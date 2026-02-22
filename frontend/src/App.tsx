import { Routes, Route, Link, useLocation } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Trades from './pages/Trades'
import Strategies from './pages/Strategies'
import Settings from './pages/Settings'
import Login from './pages/Login'

const navItems = [
  { path: '/', label: 'Dashboard' },
  { path: '/trades', label: 'Trades' },
  { path: '/strategies', label: 'Strategies' },
  { path: '/settings', label: 'Settings' },
]

function App() {
  const location = useLocation()

  if (location.pathname === '/login') {
    return (
      <Routes>
        <Route path="/login" element={<Login />} />
      </Routes>
    )
  }

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <nav className="bg-gray-800 border-b border-gray-700 px-6 py-3">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <h1 className="text-xl font-bold text-white">Trading Bot</h1>
          <div className="flex gap-4">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`px-3 py-1.5 rounded text-sm transition-colors ${
                  location.pathname === item.path
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-300 hover:text-white hover:bg-gray-700'
                }`}
              >
                {item.label}
              </Link>
            ))}
          </div>
        </div>
      </nav>
      <main className="max-w-7xl mx-auto px-6 py-6">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/trades" element={<Trades />} />
          <Route path="/strategies" element={<Strategies />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
