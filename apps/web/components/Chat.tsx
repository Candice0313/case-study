"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { ProductCard, type ProductCardData } from "./ProductCard";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const PARTSELECT_HOME = "https://www.partselect.com";
const PARTSELECT_MODELS = "https://www.partselect.com/Models/";

const PS_PART = /^PS\d{5,15}$/i;

/** Extract model number from message content (e.g. "Parts and instructions for **WRF535SWHZ**") for base-URL resolution. Excludes PartSelect part numbers (PS + digits). */
function getModelFromMessageContent(content: string | undefined): string | undefined {
  if (!content) return undefined;
  const forMatch = content.match(/for\s+\*\*([A-Z0-9][A-Z0-9\-]{4,25})\*\*/i);
  const a = forMatch ? forMatch[1].toUpperCase() : undefined;
  if (a && !PS_PART.test(a)) return a;
  const boldMatch = content.match(/\*\*([A-Z0-9][A-Z0-9\-]{4,25})\*\*/);
  const b = boldMatch ? boldMatch[1].toUpperCase() : undefined;
  return b && !PS_PART.test(b) ? b : undefined;
}

function fixContentPartSelectHomeLinks(content: string, modelFromContent?: string): string {
  if (!content) return content;
  const model = modelFromContent ?? getModelFromMessageContent(content);
  return content.replace(
    /\[([^\]]*)\]\((https:\/\/www\.partselect\.com(?:\/[^)]*)?)\)/g,
    (full, text, url) => {
      if (url.includes("/Models/") || url.includes("/Search.aspx")) return full;
      const base = text.replace(/\*\*/g, "").match(/[A-Z0-9][A-Z0-9\-]{4,24}/i)?.[0] ?? model;
      if (!base) return full;
      if (PS_PART.test(base)) return `[${text}](${PARTSELECT_HOME}/Search.aspx?SearchTerm=${encodeURIComponent(base)})`;
      return `[${text}](${PARTSELECT_MODELS}${encodeURIComponent(base.toUpperCase())}/)`;
    }
  );
}

function getTextFromChildren(children: React.ReactNode): string {
  if (children == null) return "";
  if (typeof children === "string" || typeof children === "number") return String(children);
  if (Array.isArray(children)) return children.map(getTextFromChildren).join("");
  if (typeof children === "object" && "props" in children && (children as { props?: { children?: React.ReactNode } }).props?.children != null)
    return getTextFromChildren((children as { props: { children: React.ReactNode } }).props.children);
  return "";
}

function isPartSelectHomeOnly(url: string | undefined): boolean {
  const u = (url ?? "").trim().replace(/\/$/, "");
  return u === PARTSELECT_HOME;
}

function resolvePartSelectHref(href: string | undefined, children: React.ReactNode, modelFromContent?: string): string {
  const u = (href ?? "").trim();
  if (!u.startsWith(PARTSELECT_HOME)) return u || "#";
  if (u.includes("/Models/")) return u;
  if (u.includes("/Search.aspx")) return u;
  if (!isPartSelectHomeOnly(u)) return u;
  const text = getTextFromChildren(children).replace(/\*\*/g, "");
  const base = text.match(/[A-Z0-9][A-Z0-9\-]{4,24}/i)?.[0] ?? modelFromContent;
  if (base && PS_PART.test(base)) return `${PARTSELECT_HOME}/Search.aspx?SearchTerm=${encodeURIComponent(base)}`;
  if (base) return `${PARTSELECT_MODELS}${encodeURIComponent(base.toUpperCase())}/`;
  return u || PARTSELECT_HOME + "/";
}

function partSelectSourceUrl(url: string | undefined, firstCardName?: string, modelFromContent?: string): string {
  const u = (url ?? "").trim();
  if (!u.startsWith(PARTSELECT_HOME)) return u || "#";
  if (u.includes("/Models/")) return u;
  if (u.includes("/Search.aspx")) return u;
  if (!isPartSelectHomeOnly(u)) return u;
  const base = firstCardName?.replace(/\*\*/g, "").match(/[A-Z0-9][A-Z0-9\-]{4,24}/i)?.[0] ?? modelFromContent;
  if (base && PS_PART.test(base)) return `${PARTSELECT_HOME}/Search.aspx?SearchTerm=${encodeURIComponent(base)}`;
  if (base) return `${PARTSELECT_MODELS}${encodeURIComponent(base.toUpperCase())}/`;
  return u || PARTSELECT_HOME + "/";
}

type Source = { url?: string; title?: string };
type Message =
  | { role: "user"; content: string }
  | {
      role: "assistant";
      content: string;
      product_cards?: ProductCardData[];
      sources?: Source[];
      scope_label?: string;
    };

const QUICK_ACTIONS = [
  { label: "Search by part number", prompt: "I have a part number. ", placeholder: "e.g. WP123456" },
  { label: "Search by model number", prompt: "I need parts for my appliance. Model number: ", placeholder: "e.g. WDF520PADM" },
  { label: "Describe a problem", prompt: "My appliance isn’t working right. ", placeholder: "e.g. dishwasher not filling with water" },
  { label: "Installation help", prompt: "I need installation or repair steps for: ", placeholder: "part or symptom" },
];

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [inputPlaceholder, setInputPlaceholder] = useState("Part number, model number, or describe the issue…");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendMessage(userMessage: string) {
    if (!userMessage.trim() || loading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage,
          history: messages.map((m) => ({ role: m.role, content: m.content })),
        }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`API ${res.status}: ${text.slice(0, 200)}`);
      }
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.content ?? "No response.",
          scope_label: data.scope_label,
          ...(data.product_cards?.length && { product_cards: data.product_cards }),
          ...(data.sources?.length && { sources: data.sources }),
        },
      ]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "";
      const isNetwork = !msg || msg.includes("Failed to fetch") || msg.includes("NetworkError");
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: isNetwork
            ? "Can’t reach the chat API. Make sure the backend is running (e.g. in apps/api: uvicorn main:app --reload --port 8000) and that the page is using the correct API URL."
            : `Chat error: ${msg}. Check the API is running on ${API_BASE}.`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    sendMessage(input.trim());
  }

  function onQuickAction(prompt: string, placeholder: string) {
    setInput(prompt);
    setInputPlaceholder(placeholder);
    setTimeout(() => document.querySelector<HTMLInputElement>("input[type='text']")?.focus(), 0);
  }

  return (
    <div className="flex flex-col overflow-hidden border border-[var(--ps-border)] bg-[var(--ps-white)] shadow-[var(--ps-shadow)]">
      <div
        className="border-b border-[var(--ps-border)] px-6 py-5"
        style={{ backgroundColor: "var(--ps-yellow)" }}
      >
        <h2 className="text-2xl font-bold leading-tight text-[var(--ps-text)]">
          Parts Assistant
        </h2>
        <p className="mt-1.5 text-sm leading-relaxed text-[var(--ps-text)] opacity-90">
          Refrigerators & dishwashers — find parts, check fit, get repair steps.
        </p>
      </div>

      <div className="flex max-h-[70vh] flex-col overflow-y-auto p-6">
        {messages.length === 0 && (
          <div className="space-y-5">
            <p className="text-lg font-medium text-[var(--ps-text)]">
              What do you need help with?
            </p>
            <div className="flex flex-wrap gap-3">
              {QUICK_ACTIONS.map((action, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => onQuickAction(action.prompt, action.placeholder)}
                  className="border border-[var(--ps-teal)] bg-[var(--ps-white)] px-5 py-2.5 text-base font-semibold text-[var(--ps-text)] hover:bg-[var(--ps-teal)] hover:text-white"
                >
                  {action.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div
            key={i}
            className={`mb-6 flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[90%] rounded-[var(--ps-radius)] px-5 py-4 ${
                m.role === "user"
                  ? "bg-[var(--ps-chat-user)] text-white"
                  : "bg-[var(--ps-chat-bot)] text-[var(--ps-text)]"
              }`}
            >
              <div className="text-base leading-relaxed [&_strong]:font-bold [&_p]:mb-2 [&_p:last-child]:mb-0 [&_ol]:list-decimal [&_ul]:list-disc [&_li]:ml-4 [&_ol,_ul]:my-2">
                {m.role === "assistant" ? (
                  <ReactMarkdown
                    components={{
                      a: ({ href, children, ...props }) => {
                        const resolved = resolvePartSelectHref(href, children);
                        return (
                          <a href={resolved} target="_blank" rel="noopener noreferrer" {...props}>
                            {children}
                          </a>
                        );
                      },
                    }}
                  >
                    {fixContentPartSelectHomeLinks(m.content)}
                  </ReactMarkdown>
                ) : (
                  <span className="whitespace-pre-wrap">{m.content}</span>
                )}
              </div>
              {m.role === "assistant" && m.product_cards && m.product_cards.length > 0 && (
                <div className="mt-4 space-y-3 border-t border-[var(--ps-border)] pt-4">
                  {m.product_cards.map((card, j) => (
                    <ProductCard key={j} card={card} />
                  ))}
                </div>
              )}
              {m.role === "assistant" && m.sources && m.sources.length > 0 && (
                <div className="mt-4 border-t border-[var(--ps-border)] pt-4">
                  <span className="text-sm font-semibold text-[var(--ps-text-muted)]">Source{m.sources.length !== 1 ? "s" : ""}:</span>
                  <ul className="mt-2 list-inside list-disc space-y-1 text-sm">
                    {m.sources.map((s, j) => (
                      <li key={j}>
                        <a
                          href={partSelectSourceUrl(s.url)}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-[var(--ps-teal)] underline hover:no-underline"
                        >
                          {s.title || "PartSelect"}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {m.role === "assistant" && m.scope_label === "OUT_OF_SCOPE" && (
                <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-[var(--ps-border)] pt-4">
                  <span className="text-sm font-semibold text-[var(--ps-text-muted)]">Try:</span>
                  {QUICK_ACTIONS.slice(0, 3).map((action, j) => (
                    <button
                      key={j}
                      type="button"
                      onClick={() => onQuickAction(action.prompt, action.placeholder)}
                      className="border border-[var(--ps-teal)] bg-[var(--ps-white)] px-4 py-2 text-sm font-semibold text-[var(--ps-teal)] hover:bg-[var(--ps-teal)] hover:text-white"
                    >
                      {action.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="rounded-[var(--ps-radius)] bg-[var(--ps-chat-bot)] px-5 py-4 text-base text-[var(--ps-text-muted)]">
              Looking that up…
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <form
        onSubmit={handleSubmit}
        className="border-t border-[var(--ps-border)] bg-[var(--ps-white)] p-5"
      >
        <div
          className="flex overflow-hidden"
          style={{
            border: "2px solid var(--ps-teal)",
            height: 52,
          }}
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={inputPlaceholder}
            className="min-w-0 flex-1 border-0 bg-white px-5 py-3 text-lg text-[var(--ps-text)] placeholder-[#666] focus:outline-none focus:ring-0"
            style={{ height: 52 }}
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="flex h-full shrink-0 items-center justify-center border-0 bg-[var(--ps-teal)] px-8 text-base font-bold uppercase tracking-wide text-white transition-opacity disabled:opacity-50 hover:opacity-95"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
