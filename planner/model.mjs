const DAY_MS = 86_400_000;
const PRIOR_CYCLE_DAYS = 28;

export function parseIsoDate(value) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(String(value))) return null;
  const [year, month, day] = value.split("-").map(Number);
  const date = new Date(Date.UTC(year, month - 1, day));
  if (
    date.getUTCFullYear() !== year ||
    date.getUTCMonth() !== month - 1 ||
    date.getUTCDate() !== day
  ) {
    return null;
  }
  return date;
}

export function toIsoDate(date) {
  return new Date(date).toISOString().slice(0, 10);
}

export function addDays(date, days) {
  return new Date(new Date(date).getTime() + Math.round(days) * DAY_MS);
}

export function daysBetween(earlier, later) {
  return Math.round((new Date(later).getTime() - new Date(earlier).getTime()) / DAY_MS);
}

export function normalizeStarts(values, today = new Date()) {
  const todayIso = toIsoDate(today);
  return [...new Set(values)]
    .filter((value) => parseIsoDate(value) && value <= todayIso)
    .sort();
}

export function cycleIntervals(starts) {
  const normalized = normalizeStarts(starts, new Date("9999-12-31T00:00:00Z"));
  return normalized.slice(1).map((value, index) => ({
    start: normalized[index],
    end: value,
    days: daysBetween(parseIsoDate(normalized[index]), parseIsoDate(value)),
  }));
}

export function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function weightedMean(values) {
  let weightedTotal = 0;
  let totalWeight = 0;
  values.forEach((value, index) => {
    const age = values.length - index - 1;
    const weight = 0.86 ** age;
    weightedTotal += value * weight;
    totalWeight += weight;
  });
  return weightedTotal / totalWeight;
}

function standardDeviation(values) {
  if (values.length < 2) return 0;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
    (values.length - 1);
  return Math.sqrt(variance);
}

function robustSigma(values) {
  if (values.length < 2) return 0;
  const center = median(values);
  const mad = median(values.map((value) => Math.abs(value - center)));
  return 1.4826 * mad;
}

function centerFromIntervals(intervals) {
  const recent = intervals.slice(-12);
  const bounded = recent.map((value) => Math.min(60, Math.max(15, value)));
  const personal = 0.58 * weightedMean(bounded) + 0.42 * median(bounded);
  const personalWeight = Math.min(0.94, 0.46 + 0.08 * bounded.length);
  return personalWeight * personal + (1 - personalWeight) * PRIOR_CYCLE_DAYS;
}

function backtest(intervals) {
  const errors = [];
  for (let index = 2; index < intervals.length; index += 1) {
    const predicted = centerFromIntervals(intervals.slice(0, index));
    errors.push(intervals[index] - predicted);
  }
  if (!errors.length) return null;
  const absoluteErrors = errors.map(Math.abs);
  return {
    count: errors.length,
    mae: absoluteErrors.reduce((sum, value) => sum + value, 0) / errors.length,
    withinThree:
      absoluteErrors.filter((value) => value <= 3).length / absoluteErrors.length,
    residualSigma: Math.max(robustSigma(errors), standardDeviation(errors)),
  };
}

function reliability(intervalCount, sigma, typical) {
  const coefficientOfVariation = sigma / Math.max(typical, 1);
  if (intervalCount < 2) {
    return {
      key: "early",
      label: "Early estimate",
      explanation: "Only one completed interval is available.",
    };
  }
  if (intervalCount < 4) {
    return {
      key: "learning",
      label: "Still learning",
      explanation: "More recorded starts should make the personal range more informative.",
    };
  }
  if (coefficientOfVariation > 0.2 || sigma > 7) {
    return {
      key: "variable",
      label: "Naturally broad",
      explanation: "Recorded intervals vary substantially, so a wider range is more honest.",
    };
  }
  if (intervalCount >= 6 && sigma <= 5) {
    return {
      key: "personal",
      label: "Personal pattern",
      explanation: "The estimate uses a stable pattern across several recorded intervals.",
    };
  }
  return {
    key: "developing",
    label: "Developing pattern",
    explanation: "The estimate is increasingly personalized but still uncertain.",
  };
}

export function forecastCycle(starts, today = new Date()) {
  const normalized = normalizeStarts(starts, today);
  if (normalized.length < 2) return null;

  const intervalRecords = cycleIntervals(normalized);
  const intervals = intervalRecords.map(({ days }) => days);
  const recent = intervals.slice(-12);
  const centerDays = centerFromIntervals(recent);
  const typicalDays = median(recent);
  const empiricalSigma = Math.max(
    robustSigma(recent),
    Math.min(14, standardDeviation(recent)),
  );
  const historyFloor =
    recent.length === 1 ? 5.5 : recent.length === 2 ? 4.5 : recent.length < 5 ? 3.5 : 2.5;
  const retrospective = backtest(recent);
  const sigmaDays = Math.max(
    historyFloor,
    empiricalSigma,
    retrospective ? Math.min(14, retrospective.residualSigma) : 0,
  );
  const radius50 = Math.max(2, Math.ceil(0.674 * sigmaDays));
  const radius80 = Math.max(
    recent.length === 1 ? 7 : recent.length === 2 ? 6 : recent.length < 5 ? 5 : 3,
    Math.ceil(1.282 * sigmaDays),
  );

  const lastStart = parseIsoDate(normalized.at(-1));
  const centerDate = addDays(lastStart, centerDays);
  const lower50 = addDays(centerDate, -radius50);
  const upper50 = addDays(centerDate, radius50);
  const lower80 = addDays(centerDate, -radius80);
  const upper80 = addDays(centerDate, radius80);
  const elapsedDays = daysBetween(lastStart, today);
  const expired = today.getTime() > upper80.getTime();

  return {
    starts: normalized,
    intervals: intervalRecords,
    centerDays,
    typicalDays,
    sigmaDays,
    centerDate,
    lower50,
    upper50,
    lower80,
    upper80,
    radius50,
    radius80,
    elapsedDays,
    expired,
    reliability: reliability(recent.length, sigmaDays, typicalDays),
    backtest: retrospective,
    atypicalIntervals: intervalRecords.filter(({ days }) => days < 21 || days > 35),
  };
}

export function predictionSummary(forecast) {
  if (!forecast) return null;
  return {
    center: toIsoDate(forecast.centerDate),
    likelyStart: toIsoDate(forecast.lower50),
    likelyEnd: toIsoDate(forecast.upper50),
    widerStart: toIsoDate(forecast.lower80),
    widerEnd: toIsoDate(forecast.upper80),
    reliability: forecast.reliability.label,
    expired: forecast.expired,
  };
}
