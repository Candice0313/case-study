"use client";

import { useState, useMemo } from "react";

export type ProductCardData = {
  part_number?: string;
  name?: string;
  price?: number;
  url?: string;
  image_url?: string;
  brand?: string;
};

const PARTSELECT_HOME = "https://www.partselect.com";
const PARTSELECT_MODELS = "https://www.partselect.com/Models/";

function extractModelBase(text: string): string | null {
  if (!text?.trim()) return null;
  const m = text.match(/[A-Z0-9][A-Z0-9\-]{4,24}/i);
  return m ? m[0].toUpperCase() : null;
}

function modelPageUrlFromName(name: string): string | null {
  const base = extractModelBase(name ?? "");
  return base ? PARTSELECT_MODELS + encodeURIComponent(base) + "/" : null;
}

export function ProductCard({ card }: { card: ProductCardData }) {
  const rawUrl = card.url?.trim() || "";
  const isHomepage =
    !rawUrl ||
    rawUrl === PARTSELECT_HOME ||
    rawUrl === PARTSELECT_HOME + "/" ||
    (rawUrl.startsWith(PARTSELECT_HOME) && !rawUrl.includes("/Models/"));
  const modelUrl = useMemo(
    () => (card.name && isHomepage ? modelPageUrlFromName(card.name) : null),
    [card.name, isHomepage]
  );
  const href = modelUrl || rawUrl || PARTSELECT_HOME + "/";
  const [imgError, setImgError] = useState(false);
  const showImg = card.image_url && !imgError;
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="flex gap-3 rounded-[var(--ps-radius)] border border-[var(--ps-border)] bg-[var(--ps-white)] p-3 shadow-[var(--ps-shadow)] transition hover:border-[var(--ps-teal)] hover:shadow-md"
    >
      <div className="flex h-16 w-16 shrink-0 items-center justify-center overflow-hidden border border-[var(--ps-teal)] bg-[var(--ps-off-white)]">
        {showImg ? (
          <img
            src={card.image_url}
            alt=""
            className="h-full w-full object-contain"
            referrerPolicy="no-referrer"
            onError={() => setImgError(true)}
          />
        ) : (
          <span className="text-2xl text-[var(--ps-text-muted)]">◻</span>
        )}
      </div>
      <div className="min-w-0 flex-1">
        <div className="font-medium text-[var(--ps-text)] line-clamp-2">
          {card.name || "Part"}
        </div>
        {card.part_number && (
          <div className="mt-0.5 text-sm text-[var(--ps-text-muted)]">
            Part # {card.part_number}
            {card.brand && ` · ${card.brand}`}
          </div>
        )}
        <div className="mt-2 flex items-center justify-between gap-2">
          {card.price != null && (
            <span className="font-semibold text-[var(--ps-teal)]">
              ${Number(card.price).toFixed(2)}
            </span>
          )}
          <span className="text-sm font-semibold text-[var(--ps-teal)] hover:underline">
            View on PartSelect →
          </span>
        </div>
      </div>
    </a>
  );
}
