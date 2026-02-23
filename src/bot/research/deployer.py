"""Research auto-deploy pipeline with A/B verification and safe rollback.

Evaluates research reports, applies parameter changes through the ParameterTuner,
monitors for regressions, and auto-rolls-back if performance degrades.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from bot.engines.tuner import TUNER_CONFIG, ParamChange

if TYPE_CHECKING:
    from bot.config import Settings
    from bot.engines.tracker import EngineTracker
    from bot.engines.tuner import ParameterTuner
    from bot.research.report import ResearchReport

logger = structlog.get_logger(__name__)

# Maximum number of parameters that can be changed in a single deployment
MAX_DEPLOY_CHANGES = 3


@dataclass
class DeployDecision:
    """Result of evaluating whether to deploy a research report."""

    action: str  # 'deploy' | 'skip' | 'rollback'
    reason: str
    changes: list[ParamChange] = field(default_factory=list)


@dataclass
class DeployResult:
    """Result of a deployment."""

    success: bool
    deployed_changes: list[ParamChange] = field(default_factory=list)
    snapshot_id: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "deployed_changes": [c.to_dict() for c in self.deployed_changes],
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
        }


@dataclass
class DeployRecord:
    """Record of a deployment or rollback event."""

    timestamp: str
    report_name: str
    changes: list[dict]
    snapshot_id: str
    rolled_back: bool = False
    rollback_timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "report_name": self.report_name,
            "changes": self.changes,
            "snapshot_id": self.snapshot_id,
            "rolled_back": self.rolled_back,
            "rollback_timestamp": self.rollback_timestamp,
        }


class ResearchDeployer:
    """Evaluates research reports and safely deploys parameter changes.

    Features:
    - Validates that changes are significant before deploying
    - Limits number of changes per deployment (safety bounds)
    - Ensures all changes are within TUNER_CONFIG bounds
    - Saves parameter snapshots for rollback
    - Monitors performance regression and auto-rolls-back
    """

    def __init__(
        self,
        tuner: ParameterTuner,
        settings: Settings,
        tracker: EngineTracker,
    ) -> None:
        self._tuner = tuner
        self._settings = settings
        self._tracker = tracker
        self._param_snapshots: dict[str, dict] = {}
        self._deploy_history: list[DeployRecord] = []
        # Sharpe values at deploy time for regression checking
        self._pre_deploy_sharpe: dict[str, float] = {}

    def evaluate_report(self, report: ResearchReport) -> DeployDecision:
        """Evaluate whether a research report warrants deployment.

        Only deploys when improvement_significant is True and there are
        recommended changes within safety bounds.
        """
        if not report.improvement_significant:
            return DeployDecision(
                action="skip",
                reason="Improvement not significant",
            )

        if not report.recommended_changes:
            return DeployDecision(
                action="skip",
                reason="No recommended changes",
            )

        # Filter to valid changes within TUNER_CONFIG bounds
        valid_changes = self._filter_valid_changes(report.recommended_changes)
        if not valid_changes:
            return DeployDecision(
                action="skip",
                reason="No changes within TUNER_CONFIG bounds",
            )

        # Safety bound: max N changes per deployment
        if len(valid_changes) > MAX_DEPLOY_CHANGES:
            valid_changes = valid_changes[:MAX_DEPLOY_CHANGES]
            logger.info(
                "deploy_changes_truncated",
                original=len(report.recommended_changes),
                limited=MAX_DEPLOY_CHANGES,
            )

        return DeployDecision(
            action="deploy",
            reason=f"Significant improvement with {len(valid_changes)} valid change(s)",
            changes=valid_changes,
        )

    def deploy(self, report: ResearchReport) -> DeployResult:
        """Deploy parameter changes from a research report.

        Steps:
        1. Evaluate the report
        2. Snapshot current parameters (for rollback)
        3. Apply changes via tuner
        4. Record deployment history
        """
        decision = self.evaluate_report(report)

        if decision.action != "deploy":
            logger.info(
                "deploy_skipped",
                experiment=report.experiment_name,
                reason=decision.reason,
            )
            return DeployResult(success=False)

        # Generate snapshot ID
        snapshot_id = str(uuid.uuid4())[:8]

        # Save current parameter values for rollback
        snapshot: dict[str, Any] = {}
        for change in decision.changes:
            if hasattr(self._settings, change.param_name):
                snapshot[change.param_name] = getattr(
                    self._settings, change.param_name
                )
        self._param_snapshots[snapshot_id] = snapshot

        # Record pre-deploy Sharpe for regression detection
        for change in decision.changes:
            engine = change.engine_name
            if engine not in self._pre_deploy_sharpe:
                metrics = self._tracker.get_metrics(engine, window_hours=24)
                self._pre_deploy_sharpe[engine] = metrics.sharpe_ratio

        # Apply changes
        applied = self._tuner.apply_changes(decision.changes, self._settings)

        if applied:
            record = DeployRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                report_name=report.experiment_name,
                changes=[c.to_dict() for c in decision.changes],
                snapshot_id=snapshot_id,
            )
            self._deploy_history.append(record)
            # Keep last 50 entries
            if len(self._deploy_history) > 50:
                self._deploy_history = self._deploy_history[-50:]

            logger.info(
                "deploy_success",
                experiment=report.experiment_name,
                snapshot_id=snapshot_id,
                changes=[c.to_dict() for c in decision.changes],
            )

            return DeployResult(
                success=True,
                deployed_changes=decision.changes,
                snapshot_id=snapshot_id,
            )

        return DeployResult(success=False)

    def check_regression(
        self, engine_name: str, hours_since_deploy: float = 24
    ) -> bool:
        """Check if an engine's performance has regressed since last deploy.

        Returns True if regression is detected (Sharpe dropped > 30%
        from pre-deploy value).
        """
        pre_sharpe = self._pre_deploy_sharpe.get(engine_name)
        if pre_sharpe is None:
            return False

        current_metrics = self._tracker.get_metrics(
            engine_name, window_hours=hours_since_deploy
        )
        current_sharpe = current_metrics.sharpe_ratio

        # If pre_sharpe was positive and current dropped > 30%, regression
        if pre_sharpe > 0:
            drop_pct = (pre_sharpe - current_sharpe) / pre_sharpe
            if drop_pct > 0.30:
                logger.warning(
                    "regression_detected",
                    engine=engine_name,
                    pre_sharpe=round(pre_sharpe, 4),
                    current_sharpe=round(current_sharpe, 4),
                    drop_pct=round(drop_pct * 100, 1),
                )
                return True
        elif pre_sharpe <= 0 and current_sharpe < pre_sharpe * 1.3:
            # Pre-deploy Sharpe was negative; if it got even more negative
            # by more than 30%, that's a regression
            if pre_sharpe < 0:
                drop_pct = (pre_sharpe - current_sharpe) / abs(pre_sharpe)
                if drop_pct > 0.30:
                    logger.warning(
                        "regression_detected_negative",
                        engine=engine_name,
                        pre_sharpe=round(pre_sharpe, 4),
                        current_sharpe=round(current_sharpe, 4),
                    )
                    return True

        return False

    def rollback(self, snapshot_id: str) -> bool:
        """Rollback to a saved parameter snapshot.

        Restores parameters that were saved before a deployment.
        """
        snapshot = self._param_snapshots.get(snapshot_id)
        if snapshot is None:
            logger.warning("rollback_snapshot_not_found", snapshot_id=snapshot_id)
            return False

        try:
            changed = self._settings.reload(snapshot)
            logger.info(
                "rollback_success",
                snapshot_id=snapshot_id,
                restored=changed,
            )

            # Mark the deployment as rolled back
            for record in self._deploy_history:
                if record.snapshot_id == snapshot_id and not record.rolled_back:
                    record.rolled_back = True
                    record.rollback_timestamp = datetime.now(
                        timezone.utc
                    ).isoformat()
                    break

            # Clean up pre-deploy sharpe for rolled-back engines
            # (so we don't keep checking regression on rolled-back params)
            for param_name in snapshot:
                for engine_name, config in TUNER_CONFIG.items():
                    if param_name in config:
                        self._pre_deploy_sharpe.pop(engine_name, None)

            return True
        except (ValueError, AttributeError) as e:
            logger.error("rollback_failed", snapshot_id=snapshot_id, error=str(e))
            return False

    def get_deploy_history(self) -> list[dict]:
        """Return deployment history including rollback status."""
        return [r.to_dict() for r in self._deploy_history]

    def _filter_valid_changes(
        self, changes: list[ParamChange]
    ) -> list[ParamChange]:
        """Filter changes to only those within TUNER_CONFIG bounds."""
        valid: list[ParamChange] = []
        for change in changes:
            engine_config = TUNER_CONFIG.get(change.engine_name, {})
            bounds = engine_config.get(change.param_name)
            if bounds is None:
                logger.debug(
                    "deploy_change_skipped_no_bounds",
                    engine=change.engine_name,
                    param=change.param_name,
                )
                continue

            new_val = float(change.new_value)
            # Clamp to bounds
            clamped = max(bounds.min_val, min(bounds.max_val, new_val))
            if clamped != new_val:
                # Replace with clamped value
                change = ParamChange(
                    engine_name=change.engine_name,
                    param_name=change.param_name,
                    old_value=change.old_value,
                    new_value=clamped,
                    reason=change.reason + f" (clamped to [{bounds.min_val}, {bounds.max_val}])",
                )

            valid.append(change)
        return valid
