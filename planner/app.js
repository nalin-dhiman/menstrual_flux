import {
  addDays,
  cycleIntervals,
  daysBetween,
  forecastCycle,
  normalizeStarts,
  parseIsoDate,
  predictionSummary,
  toIsoDate,
} from "./model.mjs";

const STORAGE_KEY = "menstrualFluxPlanner.v1";
const APP_VERSION = 1;
const VALID_VIEWS = new Set(["today", "history", "insights", "privacy"]);

const elements = {
  consentLayer: document.querySelector("#consent-layer"),
  adultConfirm: document.querySelector("#adult-confirm"),
  scopeConfirm: document.querySelector("#scope-confirm"),
  continueButton: document.querySelector("#continue-button"),
  greeting: document.querySelector("#greeting"),
  todayLabel: document.querySelector("#today-label"),
  predictionCard: document.querySelector("#prediction-card"),
  predictionContent: document.querySelector("#prediction-content"),
  emptyState: document.querySelector("#empty-state"),
  metricGrid: document.querySelector("#metric-grid"),
  explanationCard: document.querySelector("#explanation-card"),
  metricTypical: document.querySelector("#metric-typical"),
  metricVariation: document.querySelector("#metric-variation"),
  metricHistory: document.querySelector("#metric-history"),
  quickAddForm: document.querySelector("#quick-add-form"),
  quickDate: document.querySelector("#quick-date"),
  quickMessage: document.querySelector("#quick-message"),
  historyAddForm: document.querySelector("#history-add-form"),
  historyDate: document.querySelector("#history-date"),
  historyMessage: document.querySelector("#history-message"),
  historyCount: document.querySelector("#history-count"),
  historyList: document.querySelector("#history-list"),
  historyEmpty: document.querySelector("#history-empty"),
  intervalChart: document.querySelector("#interval-chart"),
  chartCaption: document.querySelector("#chart-caption"),
  backtestContent: document.querySelector("#backtest-content"),
  recordToday: document.querySelector("#record-today"),
  loadExample: document.querySelector("#load-example"),
  exportData: document.querySelector("#export-data"),
  requestErase: document.querySelector("#request-erase"),
  dateWarningDialog: document.querySelector("#date-warning-dialog"),
  dateWarningCopy: document.querySelector("#date-warning-copy"),
  eraseDialog: document.querySelector("#erase-dialog"),
  installDialog: document.querySelector("#install-dialog"),
  installButton: document.querySelector("#install-app"),
  toast: document.querySelector("#toast"),
};

let memoryOnly = false;
let pendingDate = null;
let deferredInstallPrompt = null;
let toastTimer = null;

function localToday() {
  const now = new Date();
  return new Date(Date.UTC(now.getFullYear(), now.getMonth(), now.getDate()));
}

function defaultState() {
  return {
    version: APP_VERSION,
    acknowledgedAt: null,
    starts: [],
    exampleMode: false,
  };
}

function readState() {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (!stored || stored.version !== APP_VERSION) return defaultState();
    return {
      ...defaultState(),
      acknowledgedAt:
        typeof stored.acknowledgedAt === "string" ? stored.acknowledgedAt : null,
      starts: normalizeStarts(
        Array.isArray(stored.starts) ? stored.starts : [],
        localToday(),
      ),
      exampleMode: Boolean(stored.exampleMode),
    };
  } catch {
    memoryOnly = true;
    return defaultState();
  }
}

let state = readState();

function persist() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    memoryOnly = false;
  } catch {
    memoryOnly = true;
    showToast("Browser storage is unavailable. Changes will last only while this page stays open.");
  }
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    elements.toast.classList.remove("visible");
  }, 3600);
}

function formatLongDate(date) {
  return new Intl.DateTimeFormat(undefined, {
    weekday: "long",
    day: "numeric",
    month: "long",
    year: "numeric",
    timeZone: "UTC",
  }).format(date);
}

function formatShortDate(date) {
  return new Intl.DateTimeFormat(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(date);
}

function formatRange(start, end) {
  if (
    start.getUTCFullYear() === end.getUTCFullYear() &&
    start.getUTCMonth() === end.getUTCMonth()
  ) {
    const monthYear = new Intl.DateTimeFormat(undefined, {
      month: "short",
      year: "numeric",
      timeZone: "UTC",
    }).format(end);
    return `${start.getUTCDate()}–${end.getUTCDate()} ${monthYear}`;
  }
  return `${formatShortDate(start)} – ${formatShortDate(end)}`;
}

function setDateLimits() {
  const todayIso = toIsoDate(localToday());
  elements.quickDate.max = todayIso;
  elements.historyDate.max = todayIso;
}

function renderHeader() {
  const hour = new Date().getHours();
  const greeting = hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";
  elements.greeting.textContent = greeting;
  elements.todayLabel.textContent = formatLongDate(localToday());
}

function makePredictionMarkup(forecast) {
  if (forecast.expired) {
    return `
      <div class="expired-card">
        <span class="kicker">The previous window has passed</span>
        <h2>Record a new start when it happens.</h2>
        <p>
          The model’s wider window ended ${formatShortDate(forecast.upper80)}.
          It has not silently moved the prediction forward. A late or missing
          period cannot be explained by this planner.
        </p>
        <button class="button" type="button" data-expired-history>
          Review recorded dates
        </button>
      </div>
    `;
  }

  const likelyRange = formatRange(forecast.lower50, forecast.upper50);
  const wideRange = formatRange(forecast.lower80, forecast.upper80);
  const exampleCopy = state.exampleMode
    ? " These are example dates; replace them before using the estimate."
    : "";
  return `
    <span class="kicker">${state.exampleMode ? "Example estimate" : "Next period planning estimate"}</span>
    <h2 class="prediction-title">
      Most likely <span class="date-emphasis">${likelyRange}</span>
    </h2>
    <p class="prediction-lead">
      Wider planning window: <strong>${wideRange}</strong>. Central estimate:
      ${formatShortDate(forecast.centerDate)}.${exampleCopy}
    </p>
    <span class="reliability-chip">
      <i aria-hidden="true"></i>
      ${forecast.reliability.label}
    </span>
    <div class="window-scale" aria-label="Visual comparison of likely and wider date windows">
      <div class="window-track">
        <span class="window-wide"></span>
        <span class="window-likely"></span>
        <span class="window-center"></span>
      </div>
      <div class="window-labels">
        <span>${formatShortDate(forecast.lower80)}</span>
        <span>central estimate</span>
        <span>${formatShortDate(forecast.upper80)}</span>
      </div>
    </div>
  `;
}

function renderPrediction() {
  const forecast = forecastCycle(state.starts, localToday());
  const hasForecast = Boolean(forecast);
  elements.predictionCard.classList.toggle("hidden", !hasForecast);
  elements.emptyState.classList.toggle("hidden", hasForecast);
  elements.metricGrid.classList.toggle("hidden", !hasForecast);
  elements.explanationCard.classList.toggle("hidden", !hasForecast);

  if (!forecast) return;
  elements.predictionContent.innerHTML = makePredictionMarkup(forecast);
  elements.metricTypical.textContent = `${Math.round(forecast.typicalDays)} days`;
  elements.metricVariation.textContent = `±${Math.round(forecast.sigmaDays)} days`;
  elements.metricHistory.textContent = `${forecast.starts.length} starts`;

  const expiredHistoryButton = elements.predictionContent.querySelector(
    "[data-expired-history]",
  );
  expiredHistoryButton?.addEventListener("click", () => showView("history"));
}

function renderHistory() {
  const starts = normalizeStarts(state.starts, localToday());
  const intervals = cycleIntervals(starts);
  const intervalByEnd = new Map(intervals.map((interval) => [interval.end, interval.days]));
  const reversed = [...starts].reverse();
  elements.historyCount.textContent = `${starts.length} ${starts.length === 1 ? "date" : "dates"}`;
  elements.historyEmpty.classList.toggle("hidden", starts.length > 0);
  elements.historyList.replaceChildren();

  reversed.forEach((value, index) => {
    const interval = intervalByEnd.get(value);
    const item = document.createElement("li");
    item.className = "history-item";

    const sequence = document.createElement("span");
    sequence.className = "history-sequence";
    sequence.textContent = String(reversed.length - index);

    const dateBlock = document.createElement("div");
    dateBlock.className = "history-date";
    const strong = document.createElement("strong");
    strong.textContent = formatShortDate(parseIsoDate(value));
    const small = document.createElement("small");
    small.textContent =
      index === 0
        ? state.exampleMode
          ? "Latest example start"
          : "Latest recorded start"
        : "Recorded start";
    dateBlock.append(strong, small);

    const intervalBadge = document.createElement("span");
    intervalBadge.className = "interval-badge";
    if (interval) {
      intervalBadge.textContent = `${interval} days`;
      if (interval < 21 || interval > 35) intervalBadge.classList.add("outside");
    } else {
      intervalBadge.textContent = "First record";
    }

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "delete-date";
    remove.dataset.deleteDate = value;
    remove.setAttribute("aria-label", `Remove ${formatShortDate(parseIsoDate(value))}`);
    remove.textContent = "×";
    remove.addEventListener("click", () => {
      state.starts = state.starts.filter((date) => date !== value);
      if (state.starts.length === 0) state.exampleMode = false;
      persist();
      renderAll();
      showToast("Date removed from this browser.");
    });

    item.append(sequence, dateBlock, intervalBadge, remove);
    elements.historyList.append(item);
  });
}

function renderChart() {
  const intervals = cycleIntervals(state.starts).slice(-12);
  elements.intervalChart.replaceChildren();
  if (!intervals.length) {
    const empty = document.createElement("div");
    empty.className = "chart-empty";
    empty.textContent = "Add at least two period starts to see an interval.";
    elements.intervalChart.append(empty);
    elements.chartCaption.textContent = "";
    return;
  }

  const scaleMaximum = Math.max(45, ...intervals.map(({ days }) => days));
  intervals.forEach(({ end, days }) => {
    const column = document.createElement("div");
    column.className = "interval-column";
    const barWrap = document.createElement("div");
    barWrap.className = "interval-bar-wrap";
    const bar = document.createElement("div");
    bar.className = "interval-bar";
    if (days < 21 || days > 35) bar.classList.add("outside");
    bar.style.height = `${Math.max(8, (days / scaleMaximum) * 165)}px`;
    bar.setAttribute("aria-label", `Interval ending ${formatShortDate(parseIsoDate(end))}: ${days} days`);
    const value = document.createElement("strong");
    value.textContent = String(days);
    const label = document.createElement("small");
    label.textContent = new Intl.DateTimeFormat(undefined, {
      month: "short",
      timeZone: "UTC",
    }).format(parseIsoDate(end));
    bar.append(value);
    barWrap.append(bar);
    column.append(barWrap, label);
    elements.intervalChart.append(column);
  });

  const outsideCount = intervals.filter(({ days }) => days < 21 || days > 35).length;
  elements.chartCaption.textContent = outsideCount
    ? `${outsideCount} recorded ${outsideCount === 1 ? "interval sits" : "intervals sit"} outside the 21–35 day reference band. This is descriptive, not a diagnosis.`
    : "All displayed intervals sit within the 21–35 day reference band.";
}

function renderBacktest() {
  const forecast = forecastCycle(state.starts, localToday());
  if (!forecast?.backtest || forecast.backtest.count < 2) {
    elements.backtestContent.innerHTML = `
      <p class="backtest-number">Learning</p>
      <p>
        At least five recorded starts are needed before the planner can compare
        several earlier estimates with what happened next.
      </p>
      <p class="small-note">
        This is a personal retrospective check, not evidence of clinical validation.
      </p>
    `;
    return;
  }

  elements.backtestContent.innerHTML = `
    <p class="backtest-number">${forecast.backtest.mae.toFixed(1)} days</p>
    <p>
      Mean absolute error across ${forecast.backtest.count} rolling personal
      back-checks. ${(forecast.backtest.withinThree * 100).toFixed(0)}% were
      within three days.
    </p>
    <p class="small-note">
      Past performance can be optimistic and does not guarantee the next cycle.
    </p>
  `;
}

function renderConsent() {
  elements.consentLayer.classList.toggle("hidden", Boolean(state.acknowledgedAt));
  document.body.style.overflow = state.acknowledgedAt ? "" : "hidden";
}

function renderStorageStatus() {
  document.querySelectorAll(".device-pill").forEach((element) => {
    element.lastChild.textContent = memoryOnly ? " Session only" : " On-device";
  });
  elements.exportData.disabled = state.starts.length === 0;
}

function renderAll() {
  state.starts = normalizeStarts(state.starts, localToday());
  renderConsent();
  renderHeader();
  renderPrediction();
  renderHistory();
  renderChart();
  renderBacktest();
  renderStorageStatus();
  setDateLimits();
}

function currentViewFromHash() {
  const candidate = window.location.hash.replace("#", "");
  return VALID_VIEWS.has(candidate) ? candidate : "today";
}

function showView(view, updateHash = true) {
  const target = VALID_VIEWS.has(view) ? view : "today";
  document.querySelectorAll(".view").forEach((section) => {
    section.classList.toggle("active", section.dataset.page === target);
  });
  document.querySelectorAll("[data-view]").forEach((button) => {
    const active = button.dataset.view === target;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  if (updateHash && window.location.hash !== `#${target}`) {
    history.replaceState(null, "", `#${target}`);
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function nearestGapFor(dateValue, starts) {
  const combined = normalizeStarts([...starts, dateValue], localToday());
  const index = combined.indexOf(dateValue);
  const gaps = [];
  if (index > 0) {
    gaps.push(daysBetween(parseIsoDate(combined[index - 1]), parseIsoDate(dateValue)));
  }
  if (index < combined.length - 1) {
    gaps.push(daysBetween(parseIsoDate(dateValue), parseIsoDate(combined[index + 1])));
  }
  return gaps.length ? Math.min(...gaps) : null;
}

function commitDate(dateValue) {
  if (state.exampleMode) {
    state.starts = [];
    state.exampleMode = false;
    showToast("Example dates were replaced with your first recorded date.");
  }
  state.starts = normalizeStarts([...state.starts, dateValue], localToday());
  persist();
  elements.quickAddForm.reset();
  elements.historyAddForm.reset();
  elements.quickMessage.textContent = "";
  elements.historyMessage.textContent = "";
  renderAll();
}

function attemptAdd(dateValue, messageElement) {
  messageElement.textContent = "";
  const date = parseIsoDate(dateValue);
  if (!date) {
    messageElement.textContent = "Choose a valid calendar date.";
    return;
  }
  if (date > localToday()) {
    messageElement.textContent = "A recorded period start cannot be in the future.";
    return;
  }
  if (state.starts.includes(dateValue) && !state.exampleMode) {
    messageElement.textContent = "That date is already recorded.";
    return;
  }

  const comparisonStarts = state.exampleMode ? [] : state.starts;
  const gap = nearestGapFor(dateValue, comparisonStarts);
  if (gap !== null && (gap < 14 || gap > 90)) {
    pendingDate = dateValue;
    elements.dateWarningCopy.textContent =
      gap < 14
        ? `This creates an interval of only ${gap} days. Make sure this was a period start rather than spotting or a duplicate entry.`
        : `This creates an interval of ${gap} days. Make sure no period start was missed between these dates.`;
    elements.dateWarningDialog.showModal();
    return;
  }
  commitDate(dateValue);
  showToast("Period start saved only in this browser.");
}

function loadExampleHistory() {
  const today = localToday();
  state.starts = [148, 120, 92, 64, 36, 8]
    .map((daysAgo) => toIsoDate(addDays(today, -daysAgo)))
    .sort();
  state.exampleMode = true;
  persist();
  renderAll();
  showToast("Example mode is active. These are not your dates.");
}

function exportPlannerData() {
  const forecast = forecastCycle(state.starts, localToday());
  const exportObject = {
    schema: "menstrual-flux-cycle-planner-export",
    version: APP_VERSION,
    exportedAt: new Date().toISOString(),
    localOnlyRelease: true,
    exampleMode: state.exampleMode,
    periodStarts: state.starts,
    intervals: cycleIntervals(state.starts),
    currentPlanningEstimate: predictionSummary(forecast),
    scope:
      "Rough period-planning estimate only; not fertility, contraception, diagnosis, or treatment.",
  };
  const blob = new Blob([`${JSON.stringify(exportObject, null, 2)}\n`], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `menstrual-flux-planner-${toIsoDate(localToday())}.json`;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  showToast("A local JSON copy was prepared.");
}

function eraseData() {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // The in-memory state is still cleared below.
  }
  state = defaultState();
  elements.adultConfirm.checked = false;
  elements.scopeConfirm.checked = false;
  elements.continueButton.disabled = true;
  renderAll();
}

function bindEvents() {
  const updateConsentButton = () => {
    elements.continueButton.disabled = !(
      elements.adultConfirm.checked && elements.scopeConfirm.checked
    );
  };
  elements.adultConfirm.addEventListener("change", updateConsentButton);
  elements.scopeConfirm.addEventListener("change", updateConsentButton);
  elements.continueButton.addEventListener("click", () => {
    if (!elements.adultConfirm.checked || !elements.scopeConfirm.checked) return;
    state.acknowledgedAt = new Date().toISOString();
    persist();
    renderConsent();
    elements.greeting.focus?.();
  });

  document.querySelectorAll("[data-view]").forEach((button) => {
    button.addEventListener("click", () => showView(button.dataset.view));
  });
  document.querySelectorAll("[data-view-jump]").forEach((button) => {
    button.addEventListener("click", () => showView(button.dataset.viewJump));
  });
  window.addEventListener("hashchange", () => showView(currentViewFromHash(), false));

  elements.quickAddForm.addEventListener("submit", (event) => {
    event.preventDefault();
    attemptAdd(elements.quickDate.value, elements.quickMessage);
  });
  elements.historyAddForm.addEventListener("submit", (event) => {
    event.preventDefault();
    attemptAdd(elements.historyDate.value, elements.historyMessage);
  });
  elements.recordToday.addEventListener("click", () => {
    attemptAdd(toIsoDate(localToday()), elements.quickMessage);
  });
  elements.loadExample.addEventListener("click", loadExampleHistory);

  elements.dateWarningDialog.addEventListener("close", () => {
    if (elements.dateWarningDialog.returnValue === "confirm" && pendingDate) {
      commitDate(pendingDate);
      showToast("Unusual interval saved. The prediction range reflects the variation.");
    }
    pendingDate = null;
  });

  elements.exportData.addEventListener("click", exportPlannerData);
  elements.requestErase.addEventListener("click", () => elements.eraseDialog.showModal());
  elements.eraseDialog.addEventListener("close", () => {
    if (elements.eraseDialog.returnValue === "erase") eraseData();
  });

  window.addEventListener("beforeinstallprompt", (event) => {
    event.preventDefault();
    deferredInstallPrompt = event;
  });
  elements.installButton.addEventListener("click", async () => {
    if (!deferredInstallPrompt) {
      elements.installDialog.showModal();
      return;
    }
    deferredInstallPrompt.prompt();
    await deferredInstallPrompt.userChoice;
    deferredInstallPrompt = null;
  });
  window.addEventListener("appinstalled", () => {
    showToast("Cycle Planner was installed on this device.");
  });

  window.addEventListener("storage", (event) => {
    if (event.key === STORAGE_KEY) {
      state = readState();
      renderAll();
      showToast("Planner data changed in another tab.");
    }
  });
}

function registerServiceWorker() {
  if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => {
      navigator.serviceWorker.register("./sw.js").catch(() => {
        // Offline installation is an enhancement; the web app remains usable.
      });
    });
  }
}

bindEvents();
renderAll();
showView(currentViewFromHash(), false);
registerServiceWorker();
