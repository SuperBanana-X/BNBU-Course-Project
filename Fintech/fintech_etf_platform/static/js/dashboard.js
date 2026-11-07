const tabs = document.querySelectorAll(".module-tab");
const views = document.querySelectorAll(".module-view");
const stateDataElement = document.getElementById("rl-state-data");
const rlStateData = stateDataElement ? JSON.parse(stateDataElement.textContent || "{}") : {};

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const target = tab.dataset.module;
    tabs.forEach((item) => item.classList.toggle("active", item === tab));
    views.forEach((view) => {
      view.classList.toggle("active", view.dataset.moduleView === target);
    });
    window.dispatchEvent(new Event("resize"));
  });
});

function normalizeTradingDate(value) {
  if (typeof value === "string") {
    return value.slice(0, 10);
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return date.toISOString().slice(0, 10);
}

function updateRlState(dateKey) {
  const state = rlStateData[dateKey];
  if (!state) {
    return;
  }

  const dateLabel = document.getElementById("selected-state-date");
  if (dateLabel) {
    dateLabel.textContent = dateKey;
  }

  Object.entries(state).forEach(([field, value]) => {
    const cell = document.querySelector(`[data-state-field="${field}"]`);
    if (cell) {
      cell.textContent = value;
    }
  });
}

function bindRlActionClick() {
  const chart = document.getElementById("rl-action-timeline");
  if (!chart || typeof chart.on !== "function") {
    window.setTimeout(bindRlActionClick, 250);
    return;
  }

  chart.on("plotly_click", (event) => {
    const point = event.points && event.points[0];
    if (!point) {
      return;
    }
    updateRlState(normalizeTradingDate(point.x));
  });
}

bindRlActionClick();
