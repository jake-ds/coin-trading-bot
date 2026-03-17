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
  login: (username: string, password: string, rememberMe?: boolean) => Promise<{ success: boolean; error?: string }>
  logout: () => void
  refreshAccessToken: () => Promise<boolean>
}

const AuthContext = createContext<AuthContextType | null>(null)

// In-memory token storage (access token never in localStorage)
let _accessToken: string | null = null
let _refreshToken: string | null = null

const RT_KEY = 'rt'

/** Get the saved refresh token from sessionStorage or localStorage */
function getSavedRefreshToken(): string | null {
  return sessionStorage.getItem(RT_KEY) || localStorage.getItem(RT_KEY)
}

/** Save refresh token — sessionStorage always, localStorage only for Remember Me */
function saveRefreshToken(token: string, rememberMe: boolean): void {
  sessionStorage.setItem(RT_KEY, token)
  if (rememberMe) {
    localStorage.setItem(RT_KEY, token)
  }
}

/** Clear refresh token from all storage */
function clearRefreshToken(): void {
  sessionStorage.removeItem(RT_KEY)
  localStorage.removeItem(RT_KEY)
}

export function getAccessToken(): string | null {
  return _accessToken
}

export function setAccessTokenDirect(token: string | null): void {
  _accessToken = token
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({
    token: null,
    refreshToken: null,
    isAuthenticated: false,
    authEnabled: null,
    username: null,
  })

  // Check auth status on mount + try to restore session
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const resp = await apiClient.get('/auth/status')
        const { auth_enabled } = resp.data

        if (!auth_enabled) {
          setState((prev) => ({
            ...prev,
            authEnabled: false,
            isAuthenticated: true,
          }))
          return
        }

        // Try to restore refresh token from storage
        const savedRT = getSavedRefreshToken()
        if (savedRT) {
          _refreshToken = savedRT
          try {
            const refreshResp = await apiClient.post('/auth/refresh', {
              refresh_token: savedRT,
            })
            const { access_token } = refreshResp.data
            _accessToken = access_token
            // Re-save to sessionStorage (in case it was only in localStorage)
            sessionStorage.setItem(RT_KEY, savedRT)
            setState({
              token: access_token,
              refreshToken: savedRT,
              isAuthenticated: true,
              authEnabled: true,
              username: null,
            })
            return
          } catch {
            // Saved refresh token is invalid — clear it
            clearRefreshToken()
            _refreshToken = null
          }
        }

        setState((prev) => ({
          ...prev,
          authEnabled: auth_enabled,
          isAuthenticated: false,
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

  const login = useCallback(async (username: string, password: string, rememberMe = false) => {
    try {
      const resp = await apiClient.post('/auth/login', {
        username,
        password,
        remember_me: rememberMe,
      })
      const { access_token, refresh_token } = resp.data
      _accessToken = access_token
      _refreshToken = refresh_token

      // Always save to sessionStorage; localStorage only for Remember Me
      saveRefreshToken(refresh_token, rememberMe)

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
    clearRefreshToken()
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
      clearRefreshToken()
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
