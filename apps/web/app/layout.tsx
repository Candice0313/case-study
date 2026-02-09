import type { Metadata } from "next";
import { Roboto } from "next/font/google";
import "./globals.css";

const roboto = Roboto({ weight: ["400", "500", "700"], subsets: ["latin"], display: "swap" });

export const metadata: Metadata = {
  title: "PartSelect – Parts Assistant",
  description: "Find refrigerator & dishwasher parts, check compatibility, get installation and repair help. PartSelect – Here to help.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={roboto.className}>
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
