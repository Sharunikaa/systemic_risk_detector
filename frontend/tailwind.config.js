/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        crisis:   '#f43f5e',
        normal:   '#10b981',
        highvol:  '#f59e0b',
        quantum:  '#818cf8',
        'bg-base':      '#0a0a0f',
        'bg-surface':   '#111118',
        'bg-elevated':  '#1a1a24',
        'bg-input':     '#1e1e2a',
        'bg-hover':     '#22222f',
        'text-primary': '#e8e8f0',
        'text-secondary':'#a0a0b8',
        'text-muted':   '#6b6b85',
        border:         '#2a2a3a',
        'border-strong':'#3a3a50',
        'border-focus': '#818cf8',
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', '"Fira Code"', 'Consolas', 'monospace'],
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      borderRadius: {
        DEFAULT: '7px',
      },
      boxShadow: {
        sm:   '0 1px 3px rgba(0,0,0,0.4)',
        md:   '0 4px 16px rgba(0,0,0,0.5)',
        glow: '0 0 20px rgba(129,140,248,0.15)',
      },
    },
  },
  plugins: [],
}
