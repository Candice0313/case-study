"use client";

import {
  LogoPartSelect,
  IconOrderStatus,
  IconUser,
  IconCart,
  IconChevronDown,
  IconSearch,
  IconDollar,
  IconTruck,
  IconOEM,
  IconWarranty,
} from "./HeaderIcons";

const PARTSELECT_SITE = "https://www.partselect.com";

const NAV_LINKS = [
  { label: "Departments", href: `${PARTSELECT_SITE}/Products/` },
  { label: "Brands", href: `${PARTSELECT_SITE}/Brands/` },
  { label: "Symptoms", href: `${PARTSELECT_SITE}/Symptoms/` },
  { label: "Blog", href: `${PARTSELECT_SITE}/Blog/` },
  { label: "Repair Help", href: `${PARTSELECT_SITE}/Repair/` },
  { label: "Water Filter Finder", href: `${PARTSELECT_SITE}/Water-Filter-Finder/` },
];

export function Header() {
  return (
    <header className="sticky top-0 z-10 bg-[var(--ps-white)]">
      {/* Top bar: Logo | Phone + hours | Order Status / Your Account / Cart (black icons) */}
      <div className="header-top-bar border-b border-[var(--ps-border)] px-4 py-2.5">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4">
          <a
            href={PARTSELECT_SITE}
            target="_blank"
            rel="noopener noreferrer"
            className="flex flex-col"
          >
            <div className="flex items-end gap-2">
              <LogoPartSelect />
              <span className="text-lg font-bold text-[var(--ps-text)] leading-none">PartSelect</span>
            </div>
            <div
              className="mt-0.5 hidden w-full px-1.5 py-0.5 text-center text-xs font-bold text-white sm:block"
              style={{ backgroundColor: "var(--ps-teal)" }}
            >
              Here to help since 1999
            </div>
          </a>

          <div className="flex items-center gap-6 text-sm">
            <div className="hidden text-center md:block">
              <a
                href="tel:1-866-319-8402"
                className="block font-bold text-[var(--ps-text)] hover:text-[var(--ps-teal)] hover:underline"
              >
                1-866-319-8402
              </a>
              <span className="text-xs text-[var(--ps-text)]">Monday to Saturday 8am - 8pm EST</span>
            </div>
            <a
              href={`${PARTSELECT_SITE}/user/self-service/`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 hover:underline"
            >
              <IconOrderStatus />
              <span>Order Status</span>
            </a>
            <a
              href={`${PARTSELECT_SITE}/user/signin/`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 hover:underline"
            >
              <IconUser />
              <span>Your Account</span>
              <IconChevronDown />
            </a>
            <a
              href={`${PARTSELECT_SITE}/shopping-cart/`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center"
              aria-label="Shopping cart"
            >
              <IconCart />
            </a>
          </div>
        </div>
      </div>

      {/* Main nav: Teal background + nav links + search box */}
      <div
        className="flex flex-wrap items-center gap-2 px-4 py-2.5"
        style={{ backgroundColor: "var(--ps-teal)" }}
      >
        <nav className="flex flex-wrap items-center gap-1 sm:gap-3">
          {NAV_LINKS.map((item) => (
            <a
              key={item.label}
              href={item.href}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-0.5 rounded px-2 py-1.5 text-sm font-bold text-white hover:bg-white/15"
            >
              {item.label}
              <IconChevronDown />
            </a>
          ))}
        </nav>
        <form
          action={`${PARTSELECT_SITE}/`}
          method="get"
          className="ml-auto flex flex-1 items-center gap-0 sm:min-w-[200px] sm:max-w-[280px]"
          role="search"
        >
          <input
            type="search"
            name="q"
            placeholder="Search model or part number"
            className="h-9 flex-1 border-0 bg-white px-2 py-1.5 text-sm text-[var(--ps-text)] placeholder-[var(--ps-text-muted)] focus:outline-none"
            style={{ borderLeft: "1px solid var(--ps-teal)" }}
          />
          <button
            type="submit"
            className="flex h-9 w-9 shrink-0 items-center justify-center bg-white text-[var(--ps-teal)] hover:opacity-90"
            style={{ border: "1px solid var(--ps-teal)", borderLeft: "none" }}
            aria-label="Search"
          >
            <IconSearch />
          </button>
        </form>
      </div>

      {/* Value props bar: Price Match / Fast Shipping / OEM / 1 Year Warranty */}
      <div
        className="flex flex-wrap items-center justify-center gap-4 px-4 py-3 text-sm font-semibold text-[var(--ps-text)]"
        style={{ backgroundColor: "var(--ps-off-white)" }}
      >
        <span className="flex items-center gap-2">
          <IconDollar />
          Price Match Guarantee
        </span>
        <span className="flex items-center gap-2">
          <IconTruck />
          Fast Shipping
        </span>
        <span className="flex items-center gap-2">
          <IconOEM />
          All Original Manufacturer Parts
        </span>
        <span className="flex items-center gap-2">
          <IconWarranty />
          1 Year Warranty
        </span>
      </div>
    </header>
  );
}
