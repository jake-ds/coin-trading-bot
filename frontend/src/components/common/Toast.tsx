import { useEffect, useState } from 'react'

export interface ToastMessage {
  id: string
  message: string
  type: 'success' | 'error' | 'info'
}

interface ToastProps {
  toasts: ToastMessage[]
  onDismiss: (id: string) => void
}

function Toast({ toasts, onDismiss }: ToastProps) {
  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 max-w-sm">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onDismiss={onDismiss} />
      ))}
    </div>
  )
}

function ToastItem({ toast, onDismiss }: { toast: ToastMessage; onDismiss: (id: string) => void }) {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    // Trigger enter animation
    requestAnimationFrame(() => setVisible(true))
    const timer = setTimeout(() => {
      setVisible(false)
      setTimeout(() => onDismiss(toast.id), 300)
    }, 5000)
    return () => clearTimeout(timer)
  }, [toast.id, onDismiss])

  const bgClass = {
    success: 'bg-green-600/90 border-green-500',
    error: 'bg-red-600/90 border-red-500',
    info: 'bg-blue-600/90 border-blue-500',
  }[toast.type]

  return (
    <div
      className={`${bgClass} border rounded-lg px-4 py-3 text-white text-sm shadow-lg transition-all duration-300 ${
        visible ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-4'
      }`}
    >
      <div className="flex items-center justify-between gap-3">
        <span>{toast.message}</span>
        <button
          onClick={() => onDismiss(toast.id)}
          className="text-white/70 hover:text-white text-lg leading-none"
        >
          &times;
        </button>
      </div>
    </div>
  )
}

export default Toast
