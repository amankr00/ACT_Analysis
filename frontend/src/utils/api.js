// Local Backend URL (Use this when running the backend locally on port 5090)
const API_BASE_URL = "http://127.0.0.1:5090";

// Cloudflare tunnel / deploy URL (uncomment and set to your tunnel hostname)
// const API_BASE_URL = "https://your-tunnel-hostname";

// All backend routes are served under the /bihar-act prefix (Cloudflare tunnel route).
const API_PREFIX = "/bihar-act";

// Polling configuration for long-running search jobs.
// The search is submitted as a job and polled until it completes, so the
// request never trips the ~100s proxy/tunnel timeout.
const POLL_INTERVAL_MS = 3000;          // poll every 3 seconds
const MAX_POLL_MS = 15 * 60 * 1000;     // give up after 15 minutes

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export async function checkServerHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}${API_PREFIX}/health`);
    if (!response.ok) throw new Error("Server health check failed");
    return await response.json();
  } catch (error) {
    console.error("Health check error:", error);
    return { status: "offline", database: "offline", error: error.message };
  }
}

export async function searchActs(statement) {
  // 1) Submit the search job — returns a job_id immediately.
  const submitResponse = await fetch(`${API_BASE_URL}${API_PREFIX}/act/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      statement: statement || "",
    }),
  });

  const submitData = await submitResponse.json();
  if (!submitResponse.ok) {
    throw new Error(submitData.message || "Act search failed.");
  }

  const jobId = submitData.job_id;
  // Defensive fallback: if the backend ever returns a result synchronously.
  if (!jobId) {
    return submitData;
  }

  // 2) Poll the status endpoint until the job completes, errors, or times out.
  const deadline = Date.now() + MAX_POLL_MS;
  while (Date.now() < deadline) {
    await sleep(POLL_INTERVAL_MS);

    const statusResponse = await fetch(
      `${API_BASE_URL}${API_PREFIX}/act/search/status/${jobId}`
    );
    const statusData = await statusResponse.json();

    if (statusResponse.ok && statusData.status === "success") {
      return statusData; // { status: "success", message, response: {...} }
    }

    if (statusData.status === "error") {
      throw new Error(statusData.message || "Act search failed.");
    }
    // status === "pending" → keep polling.
  }

  throw new Error("Search timed out after 15 minutes. Please try again.");
}

export async function triggerIngest() {
  const response = await fetch(`${API_BASE_URL}${API_PREFIX}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.message || "Ingestion failed.");
  }
  return data;
}
