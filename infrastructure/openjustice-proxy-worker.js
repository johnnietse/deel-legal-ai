// Cloudflare Worker: Proxy /api/* requests to Azure Container Instances backend
// Deployed on *.workers.dev with automatic HTTPS

const BACKEND_ORIGIN = 'http://openjustice-api.eastus2.azurecontainer.io:8000';

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, PATCH, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  'Access-Control-Max-Age': '86400',
};

export default {
  async fetch(request, env, ctx) {
    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        status: 204,
        headers: CORS_HEADERS,
      });
    }

    const url = new URL(request.url);
    const path = url.pathname + url.search;

    // Forward to ACI backend
    const backendUrl = BACKEND_ORIGIN + path;

    // Build upstream request preserving method, headers, and body
    const upstreamRequest = new Request(backendUrl, {
      method: request.method,
      headers: new Headers({
        // Strip hop-by-hop headers that shouldn't be forwarded
        ...Object.fromEntries(
          Array.from(request.headers.entries()).filter(
            ([key]) =>
              !['host', 'cf-connecting-ip', 'cf-ray', 'cf-worker', 'x-forwarded-for', 'x-real-ip', 'x-Forwarded-Proto']
                .includes(key.toLowerCase())
          )
        ),
      }),
      body: request.method !== 'GET' && request.method !== 'HEAD' ? request.body : null,
    });

    try {
      const response = await fetch(upstreamRequest);

      // Build response with CORS headers
      const responseHeaders = new Headers(response.headers);
      Object.entries(CORS_HEADERS).forEach(([key, value]) => {
        responseHeaders.set(key, value);
      });

      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: responseHeaders,
      });
    } catch (err) {
      // Backend unreachable
      return new Response(
        JSON.stringify({
          error: 'Backend unavailable',
          detail: err.message,
        }),
        {
          status: 502,
          headers: {
            'Content-Type': 'application/json',
            ...CORS_HEADERS,
          },
        }
      );
    }
  },
};
