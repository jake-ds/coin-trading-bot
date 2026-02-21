"""Main entry point for the Coin Trading Bot."""

import asyncio
import sys


async def main() -> None:
    """Run the trading bot."""
    print("Coin Trading Bot - starting...")
    print("No strategies configured yet. Exiting.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested. Exiting.")
        sys.exit(0)
