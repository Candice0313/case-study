/** @type {import('next').NextConfig} */
const path = require("path");

const nextConfig = {
  reactStrictMode: true,
  webpack: (config, { dev, isServer }) => {
    // Only manage this app's node_modules for cache (avoids root node_modules/react-icons warning)
    config.snapshot = config.snapshot || {};
    config.snapshot.managedPaths = [path.join(__dirname, "node_modules")];
    return config;
  },
};

module.exports = nextConfig;
