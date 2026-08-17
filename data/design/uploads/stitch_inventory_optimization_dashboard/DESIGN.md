---
name: Kinetic Ledger
colors:
  surface: '#131A18'
  surface-dim: '#0e1513'
  surface-bright: '#333b38'
  surface-container-lowest: '#09100e'
  surface-container-low: '#161d1b'
  surface-container: '#1a211f'
  surface-container-high: '#242b29'
  surface-container-highest: '#2f3634'
  on-surface: '#dde4e0'
  on-surface-variant: '#bccac0'
  inverse-surface: '#dde4e0'
  inverse-on-surface: '#2b3230'
  outline: '#87948b'
  outline-variant: '#3d4a43'
  surface-tint: '#69dbaa'
  primary: '#69dbaa'
  on-primary: '#003825'
  primary-container: '#2fa97c'
  on-primary-container: '#003624'
  inverse-primary: '#006c4b'
  secondary: '#a8c8ff'
  on-secondary: '#003061'
  secondary-container: '#00509a'
  on-secondary-container: '#a1c4ff'
  tertiary: '#ffb95a'
  on-tertiary: '#462a00'
  tertiary-container: '#c8892e'
  on-tertiary-container: '#432900'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#86f8c5'
  primary-fixed-dim: '#69dbaa'
  on-primary-fixed: '#002114'
  on-primary-fixed-variant: '#005138'
  secondary-fixed: '#d5e3ff'
  secondary-fixed-dim: '#a8c8ff'
  on-secondary-fixed: '#001b3c'
  on-secondary-fixed-variant: '#004689'
  tertiary-fixed: '#ffddb6'
  tertiary-fixed-dim: '#ffb95a'
  on-tertiary-fixed: '#2a1800'
  on-tertiary-fixed-variant: '#643f00'
  background: '#0e1513'
  on-background: '#dde4e0'
  surface-variant: '#2f3634'
  ground: '#0C1211'
  surface-2: '#182120'
  line-strong: '#222E2A'
  line-soft: '#1B2523'
  ink-primary: '#E8EFEC'
  ink-secondary: '#A6B5AF'
  ink-tertiary: '#7A8A84'
  jade-dim: rgba(47, 169, 124, 0.14)
  blue-dim: rgba(79, 134, 212, 0.16)
  amber-dim: rgba(196, 134, 43, 0.16)
typography:
  display-hero:
    fontFamily: JetBrains Mono
    fontSize: 52px
    fontWeight: '600'
    lineHeight: '1.1'
    letterSpacing: -0.04em
  headline-lg:
    fontFamily: Geist
    fontSize: 28px
    fontWeight: '600'
    lineHeight: '1.2'
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Geist
    fontSize: 20px
    fontWeight: '600'
    lineHeight: '1.3'
  headline-sm:
    fontFamily: Geist
    fontSize: 16px
    fontWeight: '600'
    lineHeight: '1.4'
  body-lg:
    fontFamily: Geist
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.6'
  body-md:
    fontFamily: Geist
    fontSize: 14px
    fontWeight: '400'
    lineHeight: '1.55'
  label-caps:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: '1.2'
    letterSpacing: 0.12em
  data-table:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '400'
    lineHeight: '1.4'
  headline-lg-mobile:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.2'
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  base: 8px
  xs: 4px
  sm: 12px
  md: 24px
  lg: 40px
  xl: 64px
  gutter: 24px
  margin-page: 32px
  max-width: 1280px
---

## Brand & Style

This design system is engineered for **technical authority and operational precision**. It targets supply chain analysts and inventory managers who require high-density data visualization without cognitive fatigue. The brand personality is **transparent, methodical, and expert-grade**, evoking the feel of a high-end trading terminal or scientific instrument.

The design style is a **Refined Corporate Modern** approach with a heavy emphasis on **Information Architecture**. It utilizes a dark "ground" and "surface" model to reduce eye strain during long analytical sessions. The aesthetic avoids decorative flourishes in favor of structural clarity, using precise 1px borders and purposeful color-coded semantics to guide the eye through complex datasets.

## Colors

The color palette is strictly functional, using a "Two-Pole Cost Encoding" logic to visualize trade-offs in inventory optimization.

- **Jade (Primary/Success):** Represents the "Optimum" or positive outcomes. Use for primary actions and "win" states.
- **Blue (Secondary/Info):** Represents "Understock" or missed sales. A cool-toned signal for secondary data points.
- **Amber (Tertiary/Warning):** Represents "Overstock" or spoilage. Used as a high-attention warning color for perishables and waste.
- **Surface Strategy:** The UI uses three levels of dark surfaces (`ground`, `surface`, `surface-2`) to create hierarchy. Components sit on `surface`, while secondary information or headers use `surface-2`. 
- **Neutral/Ink:** Typography uses a tiered grayscale to separate headings (`ink-primary`) from metadata (`ink-tertiary`).

## Typography

This system employs a dual-font strategy to distinguish between "Instrument" data and "Interface" controls.

- **Geist (Sans):** Used for the primary interface, headings, and instructional text. It provides a modern, neutral, and highly legible framework.
- **JetBrains Mono (Monospace):** Used for all numerical data, hero metrics, labels, and technical metadata. This reinforces the "technical instrument" aesthetic and ensures that figures in tables and charts align perfectly.
- **Tabular Numerics:** All monospace styles must use `tabular-nums` to maintain vertical alignment in financial and unit data columns.
- **Rhythm:** Aggressive letter-spacing is applied to `label-caps` (eyebrows) to distinguish them from interactive labels.

## Layout & Spacing

The layout is built on a **12-column fixed grid** with a maximum container width of `1280px`. It emphasizes a generous, structured whitespace rhythm to prevent data density from becoming overwhelming.

- **The 8px Rule:** All dimensions, padding, and margins must be multiples of 8px (e.g., 8, 16, 24, 32, 40, 64).
- **Grid Strategy:** Use a 24px gutter for standard card layouts. On desktop, cards typically span 3, 4, or 6 columns.
- **Vertical Stack:** Separate major sections with `64px` (xl) and nested components with `24px` (md).
- **Responsive Behavior:** 
  - **Desktop (1024px+):** 12 columns, 32px page margins.
  - **Tablet (768px - 1023px):** 6 columns, 24px page margins, cards reflow to 2-up.
  - **Mobile (<767px):** 4 columns, 16px page margins, cards stack vertically in a single column.

## Elevation & Depth

Hierarchy is communicated through **Tonal Layering** and **Low-Contrast Outlines**. Deep shadows and blurs are avoided to maintain the technical, flat aesthetic.

- **Base Layer:** The `ground` color (`#0C1211`) is used for the page background.
- **Surface Layer:** Interactive cards and main content areas use the `surface` color (`#131A18`) with a 1px border of `line-strong`.
- **Secondary Surface:** Inner components (like table headers or code blocks) use `surface-2` (`#182120`).
- **Interaction Depth:** On hover, cards do not lift; instead, their border color shifts to `ink-tertiary` or the primary `jade` color to indicate interactivity.
- **Indicators:** Use a 3px left-accent border for callouts and status-heavy components to provide a vertical visual anchor.

## Shapes

The shape language is **Subtle and Technical**. While the system is primarily rectangular to reinforce the grid, a small corner radius is used to soften the "ground-to-surface" transitions.

- **Standard Radius:** 4px (Soft) for all cards, buttons, and inputs.
- **Micro-Radius:** 2px for small elements like tags, chips, and data-viz legend markers.
- **Interactive Pills:** 24px+ (Pill-shaped) is reserved exclusively for primary status toggles or currency switches to distinguish them from standard buttons.

## Components

- **Cards:** The foundation of the layout. Use `surface` background, `line-strong` 1px border, and `24px` internal padding. Titles should use `headline-sm`.
- **Buttons:**
  - **Primary:** `jade` background with `ground` text. 4px radius.
  - **Secondary/Ghost:** No fill, `line-strong` border, `ink-primary` text.
- **Input Fields:** Use `ground` fill with `line-strong` border. Focus state is a 1px `jade` border with no outer glow.
- **Chips/Tags:** 2px radius, uppercase monospace font. Use "dim" background variants (e.g., `jade-dim`) for semantic meaning without overwhelming the user.
- **Data Tables:**
  - Headers: `surface-2` background, `label-caps` typography.
  - Rows: `1px solid line-soft` bottom border.
  - Cells: `data-table` typography with right-alignment for numerical units.
- **Status Callouts:** Use a 3px solid left-border (Jade/Amber/Blue) to denote the semantic category of the message.
- **Charts:** SVG line and bar marks should use a 2px stroke width. Gridlines must use `line-soft` and axis labels must use `ink-tertiary`.