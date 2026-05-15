const DATA_URL = "./data/task_006_trace.json";

const state = {
  trace: null,
  activeSection: "agentic-section",
};

const fmt = new Intl.NumberFormat("en-US");

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([key, value]) => {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;
    else node.setAttribute(key, value);
  });
  for (const child of children) {
    if (child == null) continue;
    node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

function asJson(value) {
  if (value == null || value === "") return "(empty)";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function usageBadges(usage = {}) {
  return [
    badge(`${fmt.format(usage.total_tokens || 0)} tok`),
    badge(`in ${fmt.format(usage.prompt_tokens || 0)}`),
    badge(`out ${fmt.format(usage.completion_tokens || 0)}`),
    badge(`$${Number(usage.cost || 0).toFixed(5)}`),
  ];
}

function badge(text, cls = "") {
  return el("span", { class: `badge ${cls}`, text });
}

function section(title, value) {
  return el("div", { class: "field" }, [
    el("h4", { text: title }),
    el("pre", { text: asJson(value) }),
  ]);
}

function nodeDetails({ title, type, usage, preview, children = [], body = [] }) {
  const details = el("details", { class: "tree-node", "data-search": collectSearchText({ title, preview, body }) });
  const meta = el("div", { class: "node-meta" }, [
    badge(type, type === "planner" ? "planner" : type === "subagent" ? "subagent" : type === "user_model" || type === "user" ? "user" : "tool"),
    ...(usage ? usageBadges(usage) : []),
  ]);
  const summary = el("summary", {}, [
    el("div", { class: "node-card" }, [
      el("div", { class: "node-head" }, [
        el("div", { class: "node-title" }, [
          el("span", { class: "chevron", text: "+" }),
          el("strong", { text: title }),
        ]),
        meta,
      ]),
    ]),
  ]);
  const bodyNode = el("div", { class: "node-card" }, [
    el("div", { class: "node-body" }, [
      preview ? section("Preview", preview) : null,
      ...body,
      children.length ? el("div", { class: "event-list" }, children) : null,
    ]),
  ]);
  details.append(summary, bodyNode);
  return details;
}

function collectSearchText(value) {
  return JSON.stringify(value).toLowerCase();
}

function renderSummary(trace) {
  const totals = trace.usage_totals.all;
  const summary = trace.simulation_summary;
  const items = [
    ["Reward", String(summary.reward)],
    ["Termination", summary.termination_reason],
    ["Duration", `${Number(summary.duration || 0).toFixed(1)}s`],
    ["LLM calls", fmt.format(totals.calls || 0)],
    ["Total tokens", fmt.format(totals.total_tokens || 0)],
    ["Chat cost", `$${Number(totals.cost || 0).toFixed(4)}`],
  ];
  const grid = document.getElementById("summary-grid");
  grid.replaceChildren(
    ...items.map(([label, value]) =>
      el("div", { class: "metric" }, [el("span", { text: label }), el("strong", { text: value })]),
    ),
  );
  document.getElementById("run-title").textContent = trace.result.run_id;
}

function renderObservations(trace) {
  const root = document.getElementById("observations");
  if (!root) {
    return;
  }
  root.replaceChildren(
    ...(trace.observations || []).map((item) =>
      el("div", { class: `observation ${item.level || ""}` }, [
        el("h4", { text: item.title }),
        el("p", { text: item.text }),
      ]),
    ),
  );
}

function renderAgentTree(trace) {
  const root = document.getElementById("agent-tree");
  const nodes = trace.agentic_tree.map((node) => renderAgentNode(node));
  root.replaceChildren(...nodes);
}

function renderAgentNode(node) {
  const isPlanner = node.type === "planner";
  const output = node.output || {};
  const toolCalls = output.tool_calls || [];
  const childNodes = [];

  for (const child of node.children || []) {
    childNodes.push(renderSubagentNode(child));
  }

  return nodeDetails({
    title: node.title,
    type: node.type,
    usage: node.usage,
    preview: output.content_preview || (toolCalls.length ? `Tool calls: ${toolCalls.map((t) => t.name).join(", ")}` : ""),
    children: childNodes,
    body: [
      el("div", { class: "two-col" }, [
        section(isPlanner ? "Planner input" : "User-model input", node.input_messages),
        section(isPlanner ? "Planner output" : "User-model output", output),
      ]),
    ],
  });
}

function renderSubagentNode(run) {
  const events = (run.events || []).map((event) =>
    el("div", { class: "event-row" }, [
      el("div", { class: "event-type", text: event.type }),
      el("pre", { text: event.type === "search" ? `${event.query}\n\nDocs: ${(event.doc_ids || []).join(", ")}` : asJson(event) }),
    ]),
  );
  const llmCalls = (run.llm_calls || []).map((call, index) =>
    nodeDetails({
      title: `Subagent LLM turn ${index + 1}`,
      type: "subagent",
      usage: call.usage,
      preview: call.output?.content_preview || (call.output?.tool_calls ? "Tool call" : ""),
      body: [
        el("div", { class: "two-col" }, [
          section("Input", call.input_messages),
          section("Output", call.output),
        ]),
      ],
    }),
  );
  return nodeDetails({
    title: `KB subagent ${run.number}: ${run.question}`,
    type: "subagent",
    usage: run.usage,
    preview: run.answer_preview,
    children: [...events, ...llmCalls],
    body: [
      el("div", { class: "two-col" }, [
        section("Question", run.question),
        section("Context", run.context),
      ]),
      section("Final subagent answer", run.answer),
    ],
  });
}

function renderTimeline(trace) {
  const root = document.getElementById("timeline");
  const expected = trace.task.evaluation_criteria?.actions?.[0];
  root.replaceChildren(
    ...trace.public_timeline.map((message, index) => {
      const toolNames = (message.tool_calls || []).map((tool) => tool.name);
      const isStop = (message.content || "").includes("###STOP###");
      const isExpectedTool =
        expected &&
        (message.tool_calls || []).some(
          (tool) =>
            tool.name === expected.name &&
            JSON.stringify(tool.arguments || {}) === JSON.stringify(expected.arguments || {}),
        );
      const badges = [
        badge(message.role || "message", message.role === "user" ? "user" : message.role === "assistant" ? "planner" : "tool"),
        badge(`turn ${message.turn_idx ?? index}`),
        isStop ? badge("STOP", "stop") : null,
        isExpectedTool ? badge("expected action", "subagent") : null,
        ...toolNames.map((name) => badge(name, "tool")),
      ].filter(Boolean);

      return nodeDetails({
        title: `${message.role || "message"} turn ${message.turn_idx ?? index}`,
        type: message.role || "message",
        usage: message.usage
          ? {
              prompt_tokens: message.usage.prompt_tokens || 0,
              completion_tokens: message.usage.completion_tokens || 0,
              total_tokens: (message.usage.prompt_tokens || 0) + (message.usage.completion_tokens || 0),
              cost: message.cost || 0,
            }
          : null,
        preview: message.content_preview || (toolNames.length ? `Tool calls: ${toolNames.join(", ")}` : ""),
        body: [
          el("div", { class: "node-meta" }, badges),
          el("div", { class: "two-col" }, [
            section("Content", message.content),
            section("Tool calls", message.tool_calls),
          ]),
        ],
      });
    }),
  );
}

function wireTabs() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("active"));
      document.querySelectorAll(".trace-section").forEach((section) => section.classList.remove("active-section"));
      button.classList.add("active");
      document.getElementById(button.dataset.target).classList.add("active-section");
    });
  });
}

function wireExpandCollapse() {
  document.getElementById("expand-all").addEventListener("click", () => {
    document.querySelectorAll("details").forEach((item) => (item.open = true));
  });
  document.getElementById("collapse-all").addEventListener("click", () => {
    document.querySelectorAll("details").forEach((item) => (item.open = false));
  });
}

function wireFilter() {
  const input = document.getElementById("trace-filter");
  input.addEventListener("input", () => {
    const query = input.value.trim().toLowerCase();
    document.querySelectorAll("details.tree-node").forEach((node) => {
      if (!query) {
        node.classList.remove("hidden-by-filter");
        return;
      }
      const haystack = node.getAttribute("data-search") || "";
      node.classList.toggle("hidden-by-filter", !haystack.includes(query));
    });
  });
}

async function main() {
  const response = await fetch(DATA_URL);
  const trace = await response.json();
  state.trace = trace;
  renderSummary(trace);
  renderAgentTree(trace);
  wireExpandCollapse();
  wireFilter();
}

main().catch((error) => {
  document.body.innerHTML = `<pre style="padding:20px;color:#b42318">Failed to load dashboard: ${error.stack || error}</pre>`;
});
