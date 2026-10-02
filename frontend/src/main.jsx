import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'
import './api' // 给后端请求带上界面语言（副作用引入，见 api.js）

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
