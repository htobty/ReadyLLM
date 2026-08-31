// 「ReadyLLM」品牌 Logo —— 纯矢量 SVG，自包含，可任意缩放
// 设计：深色渐变圆角底 + 渐变环形进度弧（就绪/加载）+ 中心播放三角（一键部署运行）+ 绿色状态点（在线）
// size 控制整体像素尺寸；渐变 id 用 rdl- 前缀避免与页面其它 SVG 冲突

export default function Logo({ size = 36, className = '' }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 128 128"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      role="img"
      aria-label="ReadyLLM"
    >
      <defs>
        <linearGradient id="rdl-bg" x1="0" y1="0" x2="128" y2="128" gradientUnits="userSpaceOnUse">
          <stop stopColor="#0b1220" />
          <stop offset="1" stopColor="#16213e" />
        </linearGradient>
        <linearGradient id="rdl-ring" x1="0" y1="0" x2="128" y2="128" gradientUnits="userSpaceOnUse">
          <stop stopColor="#22d3ee" />
          <stop offset="0.5" stopColor="#3b82f6" />
          <stop offset="1" stopColor="#8b5cf6" />
        </linearGradient>
      </defs>

      {/* 深色渐变圆角底 */}
      <rect x="4" y="4" width="120" height="120" rx="28" fill="url(#rdl-bg)" />

      {/* 渐变环形进度弧：表达「就绪 / 加载」 */}
      <circle
        cx="64"
        cy="64"
        r="34"
        fill="none"
        stroke="url(#rdl-ring)"
        strokeWidth="8"
        strokeLinecap="round"
        strokeDasharray="163 51"
        transform="rotate(-120 64 64)"
      />

      {/* 中心播放三角：一键部署 / 运行 */}
      <path d="M56 50 L82 64 L56 78 Z" fill="#ffffff" />

      {/* 绿色状态点：在线 / 就绪 */}
      <circle cx="88" cy="40" r="6.5" fill="#34d399" />
    </svg>
  )
}
