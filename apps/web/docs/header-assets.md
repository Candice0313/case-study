# 顶部 Logo 与图标从哪里找

## PartSelect 品牌 Logo（橙色房子 + P）

- **官方来源**：PartSelect 官网使用的 logo 为品牌资产，正式项目应向 PartSelect 品牌或设计团队索取（如 `ps-logo-mobile.svg`、`ps-header-logo-here-to-help.svg`）。
- **官网 CDN 参考**：  
  `https://partselectcom-gtcdcddbene3cpes.z01.azurefd.net/images/ps-logo-mobile.svg`  
  （仅供开发参考，上线请使用自有授权资源。）
- **当前实现**：`components/HeaderIcons.tsx` 中的 `LogoPartSelect` 为占位用内联 SVG，可替换为 `<img src="/images/ps-logo.svg" alt="PartSelect" />` 或 Next.js `Image`。

---

## 通用 UI 图标（Order Status、账户、购物车、搜索、chevron 等）

可从以下图标库选用并导出 SVG 或使用 React 组件：

| 来源 | 说明 |
|------|------|
| **Lucide** | https://lucide.dev — 开源 SVG 图标，风格简洁，可直接复制 SVG 或使用 `lucide-react`。 |
| **Heroicons** | https://heroicons.com — Tailwind 官方，提供 React/SVG。 |
| **Material Icons** | https://fonts.google.com/icons — 谷歌 Material 图标。 |
| **Font Awesome** | https://fontawesome.com — 需注册，部分免费；可下载 SVG。 |

当前项目在 `HeaderIcons.tsx` 中用**内联 SVG** 实现了上述图标，无需额外安装；若要统一风格，可改为从上述任选一个库引入并替换。

---

## 价值主张条图标（$、卡车、齿轮、盾牌/勋章）

- **Price Match（美元）**、**Fast Shipping（卡车）**、**OEM（齿轮/勾）**、**1 Year Warranty（盾牌/勋章）** 在图标库中多为通用图标，同上表（Lucide / Heroicons / Material）搜索 "dollar"、"truck"、"settings/cog"、"shield/award" 即可。
- 若官网使用**定制组合图标**（如齿轮里带勾），可：在图标库中找相近图标叠加，或向品牌方要设计稿/导出的 SVG。

---

## 替换为自有资源的方式

1. **PartSelect Logo**  
   将官方提供的 SVG/PNG 放到 `apps/web/public/images/`，在 `Header.tsx` 里用：
   ```tsx
   <Image src="/images/ps-header-logo.svg" alt="PartSelect" width={…} height={…} />
   ```
   或 `<img src="/images/ps-header-logo.svg" alt="PartSelect" />`，并移除或隐藏当前的 `LogoPartSelect` 占位。

2. **图标**  
   若使用图标库（如 `lucide-react`）：安装后在 `Header.tsx` 或 `HeaderIcons.tsx` 中引入对应组件，替换现有内联 SVG 即可。
