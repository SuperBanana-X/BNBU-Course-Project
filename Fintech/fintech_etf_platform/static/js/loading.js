const steps = [
  ["Initializing quantitative research workstation...", 0],
  ["Connecting to AkShare market data API...", 25],
  ["Loading ETF OHLCV data...", 50],
  ["Generating technical feature layer...", 70],
  ["Connecting to encrypted database...", 85],
  ["Loading sentiment feature layer...", 85],
  ["Loading RL decision engine...", 85],
  ["Loading portfolio performance records...", 100],
  ["Finalizing dashboard...", 100],
];

const bar = document.getElementById("progress-bar");
const value = document.getElementById("progress-value");
const stepLabel = document.getElementById("loading-step");
const log = document.getElementById("loading-log");

steps.forEach((step, index) => {
  window.setTimeout(() => {
    const [label, progress] = step;
    stepLabel.textContent = label;
    value.textContent = `${progress}%`;
    bar.style.width = `${progress}%`;
    const item = document.createElement("li");
    item.textContent = label;
    log.appendChild(item);
  }, index * 760);
});

window.setTimeout(() => {
  window.location.href = "/dashboard";
}, 7000);
