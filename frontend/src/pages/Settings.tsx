import { useState, useEffect, useCallback } from 'react'
import apiClient from '../api/client'
import type { SettingItem, SettingsResponse, SettingsUpdateResponse } from '../api/types'

const SECTION_ORDER = [
  'Trading',
  'Risk Management',
  'Strategies',
  'Exchange',
  'Dashboard',
  'Notifications',
]

function Settings() {
  const [settings, setSettings] = useState<SettingItem[]>([])
  const [editValues, setEditValues] = useState<Record<string, unknown>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [saveMessage, setSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [showConfirm, setShowConfirm] = useState(false)
  const [previousValues, setPreviousValues] = useState<Record<string, unknown> | null>(null)

  const fetchSettings = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await apiClient.get<SettingsResponse>('/settings')
      setSettings(response.data.settings)
      // Initialize edit values from current settings
      const vals: Record<string, unknown> = {}
      for (const s of response.data.settings) {
        vals[s.key] = s.value
      }
      setEditValues(vals)
    } catch {
      setError('Failed to load settings')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchSettings()
  }, [fetchSettings])

  // Get changed fields (only safe, non-restart fields)
  const getChangedFields = useCallback(() => {
    const changed: Record<string, unknown> = {}
    for (const s of settings) {
      if (s.requires_restart) continue
      if (s.type === 'secret') continue
      const editVal = editValues[s.key]
      const origVal = s.value
      if (JSON.stringify(editVal) !== JSON.stringify(origVal)) {
        changed[s.key] = editVal
      }
    }
    return changed
  }, [settings, editValues])

  const handleSave = useCallback(() => {
    const changed = getChangedFields()
    if (Object.keys(changed).length === 0) {
      setSaveMessage({ type: 'error', text: 'No changes to save' })
      setTimeout(() => setSaveMessage(null), 3000)
      return
    }
    setShowConfirm(true)
  }, [getChangedFields])

  const confirmSave = useCallback(async () => {
    setShowConfirm(false)
    const changed = getChangedFields()
    if (Object.keys(changed).length === 0) return

    try {
      setSaving(true)
      const response = await apiClient.put<SettingsUpdateResponse>('/settings', changed)
      setPreviousValues(response.data.previous)
      setSaveMessage({
        type: 'success',
        text: `Updated ${response.data.changed.length} setting(s): ${response.data.changed.join(', ')}`,
      })
      // Refresh settings from server
      await fetchSettings()
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setSaveMessage({ type: 'error', text: detail || 'Failed to save settings' })
    } finally {
      setSaving(false)
      setTimeout(() => setSaveMessage(null), 5000)
    }
  }, [getChangedFields, fetchSettings])

  const handleUndo = useCallback(async () => {
    if (!previousValues) return
    try {
      setSaving(true)
      const response = await apiClient.put<SettingsUpdateResponse>('/settings', previousValues)
      setPreviousValues(null)
      setSaveMessage({
        type: 'success',
        text: `Reverted ${response.data.changed.length} setting(s)`,
      })
      await fetchSettings()
    } catch {
      setSaveMessage({ type: 'error', text: 'Failed to undo changes' })
    } finally {
      setSaving(false)
      setTimeout(() => setSaveMessage(null), 5000)
    }
  }, [previousValues, fetchSettings])

  const updateEditValue = useCallback((key: string, value: unknown) => {
    setEditValues((prev) => ({ ...prev, [key]: value }))
  }, [])

  if (loading) {
    return (
      <div>
        <h2 className="text-2xl font-bold mb-6">Settings</h2>
        <div className="space-y-6">
          {[1, 2, 3].map((i) => (
            <div key={i} className="bg-gray-800 rounded-lg p-6 animate-pulse">
              <div className="h-6 bg-gray-700 rounded w-32 mb-4" />
              <div className="space-y-3">
                <div className="h-10 bg-gray-700 rounded" />
                <div className="h-10 bg-gray-700 rounded" />
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div>
        <h2 className="text-2xl font-bold mb-6">Settings</h2>
        <div className="bg-red-900/30 border border-red-700 rounded-lg p-4 text-red-300">
          {error}
        </div>
      </div>
    )
  }

  // Group settings by section
  const grouped: Record<string, SettingItem[]> = {}
  for (const s of settings) {
    const section = s.section || 'Other'
    if (!grouped[section]) grouped[section] = []
    grouped[section].push(s)
  }

  const orderedSections = SECTION_ORDER.filter((s) => grouped[s])
  // Append any unlisted sections
  for (const s of Object.keys(grouped)) {
    if (!orderedSections.includes(s)) orderedSections.push(s)
  }

  const changedFields = getChangedFields()
  const hasChanges = Object.keys(changedFields).length > 0

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold">Settings</h2>
        <div className="flex items-center gap-3">
          {previousValues && (
            <button
              onClick={handleUndo}
              disabled={saving}
              className="px-4 py-2 rounded bg-gray-600 hover:bg-gray-500 text-white text-sm transition-colors disabled:opacity-50"
            >
              Undo
            </button>
          )}
          <button
            onClick={handleSave}
            disabled={saving || !hasChanges}
            className="px-4 py-2 rounded bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium transition-colors disabled:opacity-50"
          >
            {saving ? 'Saving...' : 'Save Changes'}
          </button>
        </div>
      </div>

      {/* Save message */}
      {saveMessage && (
        <div
          className={`mb-4 rounded-lg p-3 text-sm ${
            saveMessage.type === 'success'
              ? 'bg-green-900/30 border border-green-700 text-green-300'
              : 'bg-red-900/30 border border-red-700 text-red-300'
          }`}
        >
          {saveMessage.text}
        </div>
      )}

      {/* Confirmation dialog */}
      {showConfirm && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4 border border-gray-600">
            <h3 className="text-lg font-bold mb-3">Confirm Changes</h3>
            <p className="text-gray-300 text-sm mb-4">The following settings will be updated:</p>
            <ul className="text-sm space-y-1 mb-6">
              {Object.entries(changedFields).map(([key, val]) => (
                <li key={key} className="text-yellow-300">
                  <span className="text-gray-400">{key}:</span> {JSON.stringify(val)}
                </li>
              ))}
            </ul>
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setShowConfirm(false)}
                className="px-4 py-2 rounded bg-gray-600 hover:bg-gray-500 text-white text-sm"
              >
                Cancel
              </button>
              <button
                onClick={confirmSave}
                className="px-4 py-2 rounded bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium"
              >
                Apply
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Sections */}
      <div className="space-y-6">
        {orderedSections.map((section) => (
          <div key={section} className="bg-gray-800 rounded-lg border border-gray-700 p-6">
            <h3 className="text-lg font-semibold mb-4 text-white">{section}</h3>
            <div className="space-y-4">
              {grouped[section].map((s) => (
                <SettingRow
                  key={s.key}
                  setting={s}
                  editValue={editValues[s.key]}
                  onChange={updateEditValue}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

interface SettingRowProps {
  setting: SettingItem
  editValue: unknown
  onChange: (key: string, value: unknown) => void
}

function SettingRow({ setting, editValue, onChange }: SettingRowProps) {
  const isReadOnly = setting.requires_restart || setting.type === 'secret'
  const isSecret = setting.type === 'secret'

  return (
    <div className="flex flex-col sm:flex-row sm:items-center gap-2 py-2 border-b border-gray-700/50 last:border-0">
      <div className="sm:w-1/3">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-200">{setting.key}</span>
          {setting.requires_restart && (
            <span className="text-xs px-1.5 py-0.5 rounded bg-yellow-900/50 text-yellow-400 border border-yellow-700/50">
              restart
            </span>
          )}
        </div>
        <p className="text-xs text-gray-500 mt-0.5">{setting.description}</p>
      </div>
      <div className="sm:w-2/3">
        <SettingInput
          setting={setting}
          value={editValue}
          onChange={(val) => onChange(setting.key, val)}
          readOnly={isReadOnly}
          masked={isSecret}
        />
      </div>
    </div>
  )
}

interface SettingInputProps {
  setting: SettingItem
  value: unknown
  onChange: (val: unknown) => void
  readOnly: boolean
  masked: boolean
}

function SettingInput({ setting, value, onChange, readOnly, masked }: SettingInputProps) {
  const baseClasses =
    'w-full px-3 py-2 rounded bg-gray-700 border border-gray-600 text-sm text-gray-200 focus:outline-none focus:border-blue-500'
  const readOnlyClasses = readOnly ? 'opacity-60 cursor-not-allowed' : ''

  if (setting.type === 'bool') {
    return (
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(e) => onChange(e.target.checked)}
          disabled={readOnly}
          className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-blue-500 focus:ring-blue-500 disabled:opacity-60"
        />
        <span className="text-sm text-gray-300">{value ? 'Enabled' : 'Disabled'}</span>
      </label>
    )
  }

  if (setting.type === 'select' && setting.options) {
    return (
      <select
        value={String(value || '')}
        onChange={(e) => onChange(e.target.value)}
        disabled={readOnly}
        className={`${baseClasses} ${readOnlyClasses}`}
      >
        {setting.options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    )
  }

  if (masked) {
    return (
      <input
        type="text"
        value={String(value || '')}
        readOnly
        className={`${baseClasses} opacity-60 cursor-not-allowed`}
      />
    )
  }

  if (setting.type === 'int' || setting.type === 'float') {
    return (
      <input
        type="number"
        value={value === undefined || value === null ? '' : String(value)}
        onChange={(e) => {
          const v = e.target.value
          if (v === '') {
            onChange(setting.type === 'int' ? 0 : 0.0)
          } else {
            onChange(setting.type === 'int' ? parseInt(v, 10) : parseFloat(v))
          }
        }}
        step={setting.type === 'float' ? '0.1' : '1'}
        readOnly={readOnly}
        className={`${baseClasses} ${readOnlyClasses}`}
      />
    )
  }

  if (setting.type === 'list' || setting.type === 'dict') {
    return (
      <input
        type="text"
        value={typeof value === 'string' ? value : JSON.stringify(value)}
        onChange={(e) => {
          try {
            onChange(JSON.parse(e.target.value))
          } catch {
            // Keep raw string while user is typing
          }
        }}
        readOnly={readOnly}
        className={`${baseClasses} ${readOnlyClasses}`}
      />
    )
  }

  // Default: string input
  return (
    <input
      type="text"
      value={String(value || '')}
      onChange={(e) => onChange(e.target.value)}
      readOnly={readOnly}
      className={`${baseClasses} ${readOnlyClasses}`}
    />
  )
}

export default Settings
