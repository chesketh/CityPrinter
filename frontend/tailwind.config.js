/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        ctp: {
          base:      '#1e1e2e',
          mantle:    '#181825',
          crust:     '#11111b',
          surface0:  '#313244',
          surface1:  '#45475a',
          surface2:  '#585b70',
          overlay0:  '#6c7086',
          overlay1:  '#7f849c',
          overlay2:  '#9399b2',
          subtext0:  '#a6adc8',
          subtext1:  '#bac2de',
          text:      '#cdd6f4',
          blue:      '#89b4fa',
          green:     '#a6e3a1',
          red:       '#f38ba8',
          yellow:    '#f9e2af',
          peach:     '#fab387',
          mauve:     '#cba6f7',
          lavender:  '#b4befe',
          sky:       '#89dceb',
          teal:      '#94e2d5',
        },
      },
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', "'Segoe UI'", 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      animation: {
        shimmer:      'shimmer 1.5s ease-in-out infinite',
        'fade-in':    'fadeIn 0.2s ease-out',
        'slide-down': 'slideDown 0.15s ease-out',
      },
      keyframes: {
        shimmer: {
          '0%':   { backgroundPosition: '200% 0' },
          '100%': { backgroundPosition: '-200% 0' },
        },
        fadeIn: {
          '0%':   { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideDown: {
          '0%':   { opacity: '0', transform: 'translateY(-4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}
