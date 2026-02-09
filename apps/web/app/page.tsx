"use client";

import { Header } from "@/components/Header";
import { Chat } from "@/components/Chat";

const PARTSELECT_SITE = "https://www.partselect.com";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="flex-1 px-4 py-4" style={{ backgroundColor: "var(--ps-white)" }}>
        <div className="mx-auto w-full max-w-4xl">
          <Chat />

          <footer className="mt-8 text-center text-sm text-[var(--ps-text)]">
            <p>
              Need help finding your model number?{" "}
              <a
                href={`${PARTSELECT_SITE}/Find-Your-Model-Number/`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-[var(--ps-teal)] hover:underline"
              >
                Use our model number locator
              </a>
            </p>
            <p className="mt-1">
              Mon–Sat 8am–8pm EST:{" "}
              <a href="tel:1-866-319-8402" className="text-[var(--ps-teal)] hover:underline">
                1-866-319-8402
              </a>
            </p>
          </footer>
        </div>
      </main>
    </div>
  );
}
