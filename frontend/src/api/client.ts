import axios from 'axios'
import type { AxiosError, InternalAxiosRequestConfig } from 'axios'
import { getAccessToken, setAccessTokenDirect } from '../hooks/useAuth'

const apiClient = axios.create({
  baseURL: '/api',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add auth token to requests
apiClient.interceptors.request.use((config) => {
  const token = getAccessToken()
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Track if we're already refreshing to avoid infinite loops
let isRefreshing = false
let failedQueue: Array<{
  resolve: (value: unknown) => void
  reject: (error: unknown) => void
  config: InternalAxiosRequestConfig
}> = []

function processQueue(error: unknown) {
  failedQueue.forEach(({ reject }) => reject(error))
  failedQueue = []
}

// Handle 401 responses — auto refresh then retry
apiClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config
    if (!originalRequest) return Promise.reject(error)

    // Don't retry auth endpoints
    const url = originalRequest.url || ''
    if (url.startsWith('/auth/')) {
      return Promise.reject(error)
    }

    if (error.response?.status === 401) {
      if (isRefreshing) {
        // Queue this request while refresh is in progress
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject, config: originalRequest })
        })
      }

      isRefreshing = true

      // Try to refresh using saved refresh token (sessionStorage first, then localStorage)
      const savedRT = sessionStorage.getItem('rt') || localStorage.getItem('rt')
      if (savedRT) {
        try {
          const resp = await axios.post('/api/auth/refresh', {
            refresh_token: savedRT,
          })
          const { access_token } = resp.data
          setAccessTokenDirect(access_token)

          // Retry all queued requests
          failedQueue.forEach(({ resolve, config }) => {
            config.headers.Authorization = `Bearer ${access_token}`
            resolve(apiClient(config))
          })
          failedQueue = []
          isRefreshing = false

          // Retry original request
          originalRequest.headers.Authorization = `Bearer ${access_token}`
          return apiClient(originalRequest)
        } catch {
          processQueue(error)
          isRefreshing = false
          localStorage.removeItem('rt')
          window.location.href = '/login'
          return Promise.reject(error)
        }
      }

      isRefreshing = false
      window.location.href = '/login'
    }
    return Promise.reject(error)
  },
)

export default apiClient
