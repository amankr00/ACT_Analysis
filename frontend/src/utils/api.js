// Local Backend URL (Use this when running the backend locally)
// const API_BASE_URL = "http://127.0.0.1:8080";

// Deploy URL (Commented out)
const API_BASE_URL = "https://act-analysis.onrender.com";

export async function checkServerHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/judicialAI-health`);
    if (!response.ok) throw new Error("Server health check failed");
    return await response.json();
  } catch (error) {
    console.error("Health check error:", error);
    return { status: "offline", database: "offline", error: error.message };
  }
}

export async function searchActs(statement) {
  const response = await fetch(`${API_BASE_URL}/judicialAI/act/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      statement: statement || "",
    }),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.message || "Act search failed.");
  }
  return data;
}

export async function triggerIngest() {
  const response = await fetch(`${API_BASE_URL}/judicialAI/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.message || "Ingestion failed.");
  }
  return data;
}
