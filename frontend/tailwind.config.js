/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#14141f',
        card: '#22222f',
        fg: '#cdd6f4',
        green: '#a6e3a1',
        red: '#f38ba8',
        yellow: '#f9e2af',
        blue: '#89b4fa',
        purple: '#cba6f7',
        teal: '#94e2d5',
        orange: '#fab387',
        gray: '#6c7086',
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'PingFang SC', 'sans-serif'],
        mono: ['JetBrains Mono', 'SF Mono', 'ui-monospace', 'Menlo', 'monospace'],
      },
      borderRadius: {
        xl: '0.875rem',
        '2xl': '1.125rem',
      },
      boxShadow: {
        card: '0 12px 32px -16px rgba(0,0,0,0.6), 0 1px 0 rgba(255,255,255,0.03) inset',
        glow: '0 0 0 1px rgba(137,180,250,0.25), 0 8px 24px -8px rgba(137,180,250,0.35)',
        'glow-green': '0 0 0 1px rgba(166,227,161,0.25), 0 8px 24px -8px rgba(166,227,161,0.35)',
      },
      keyframes: {
        fadeIn: { from: { opacity: '0' }, to: { opacity: '1' } },
        slideUp: {
          from: { opacity: '0', transform: 'translateY(10px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        'fade-in': 'fadeIn 0.4s ease both',
        'slide-up': 'slideUp 0.45s cubic-bezier(0.22,1,0.36,1) both',
      },
    },
  },
  plugins: [],
}
