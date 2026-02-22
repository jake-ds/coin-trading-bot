import { useState, useEffect, useCallback, createContext, useContext } from 'react'
import type { ReactNode } from 'react'
import { createElement } from 'react'
import apiClient from '../api/client'

interface AuthState {
  token: string | null
  refreshToken: string | null
  isAuthenticated: boolean
  authEnabled: boolean | null // null = loading
  username: string | null
}

interface AuthContextType extends AuthState {
  login: (username: string, password: string) => Promise<{ success: boolean; error?: string }>
  logout: () => void
  refreshAccessToken: () => Promise<boolean>
}

const AuthContext = createContext<AuthContextType | null>(null)

// In-memory token storage (not localStorage for security)
let _accessToken: string | null = null
let _refreshToken: string | null = null

export function getAccessToken(): string | null {
  return _accessToken
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({
    token: null,
    refreshToken: null,
    isAuthenticated: false,
    authEnabled: null,
    username: null,
  })

  // Check auth status on mount
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const resp = await apiClient.get('/auth/status')
        const { auth_enabled } = resp.data
        setState((prev) => ({
          ...prev,
          authEnabled: auth_enabled,
          // If auth is disabled, consider user authenticated
          isAuthenticated: !auth_enabled,
        }))
      } catch {
        // If we can't reach the server, assume auth is disabled
        setState((prev) => ({
          ...prev,
          authEnabled: false,
          isAuthenticated: true,
        }))
      }
    }
    checkAuth()
  }, [])

  const login = useCallback(async (username: string, password: string) => {
    try {
      const resp = await apiClient.post('/auth/login', { username, password })
      const { access_token, refresh_token } = resp.data
      _accessToken = access_token
      _refreshToken = refresh_token
      setState({
        token: access_token,
        refreshToken: refresh_token,
        isAuthenticated: true,
        authEnabled: true,
        username,
      })
      return { success: true }
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } } }
      return {
        success: false,
        error: error.response?.data?.detail || 'Login failed',
      }
    }
  }, [])

  const logout = useCallback(async () => {
    if (_refreshToken) {
      try {
        await apiClient.post('/auth/logout', { refresh_token: _refreshToken })
      } catch {
        // Ignore errors during logout
      }
    }
    _accessToken = null
    _refreshToken = null
    setState((prev) => ({
      ...prev,
      token: null,
      refreshToken: null,
      isAuthenticated: false,
      username: null,
    }))
  }, [])

  const refreshAccessToken = useCallback(async () => {
    if (!_refreshToken) return false
    try {
      const resp = await apiClient.post('/auth/refresh', {
        refresh_token: _refreshToken,
      })
      const { access_token } = resp.data
      _accessToken = access_token
      setState((prev) => ({
        ...prev,
        token: access_token,
      }))
      return true
    } catch {
      // Refresh failed — force logout
      _accessToken = null
      _refreshToken = null
      setState((prev) => ({
        ...prev,
        token: null,
        refreshToken: null,
        isAuthenticated: false,
        username: null,
      }))
      return false
    }
  }, [])

  const value: AuthContextType = {
    ...state,
    login,
    logout,
    refreshAccessToken,
  }

  return createElement(AuthContext.Provider, { value }, children)
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
