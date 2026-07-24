import test from "node:test";
import assert from "node:assert/strict";

import {
  addDays,
  cycleIntervals,
  daysBetween,
  forecastCycle,
  normalizeStarts,
  parseIsoDate,
  predictionSummary,
  toIsoDate,
} from "../../docs/site/planner/model.mjs";

test("date helpers use calendar days without timezone drift", () => {
  const start = parseIsoDate("2026-01-31");
  assert.equal(toIsoDate(addDays(start, 28)), "2026-02-28");
  assert.equal(daysBetween(start, parseIsoDate("2026-03-02")), 30);
  assert.equal(parseIsoDate("2026-02-30"), null);
});

test("start dates are deduplicated, ordered, and future dates are removed", () => {
  assert.deepEqual(
    normalizeStarts(
      ["2026-04-01", "bad", "2026-03-01", "2026-03-01", "2027-01-01"],
      new Date("2026-04-15T12:00:00Z"),
    ),
    ["2026-03-01", "2026-04-01"],
  );
});

test("cycle intervals retain actual recorded differences", () => {
  assert.deepEqual(
    cycleIntervals(["2026-01-01", "2026-01-29", "2026-02-27"]).map(
      ({ days }) => days,
    ),
    [28, 29],
  );
});

test("regular history produces a narrow personalized forecast", () => {
  const forecast = forecastCycle(
    [
      "2026-01-01",
      "2026-01-29",
      "2026-02-26",
      "2026-03-26",
      "2026-04-23",
      "2026-05-21",
      "2026-06-18",
      "2026-07-16",
    ],
    new Date("2026-07-24T12:00:00Z"),
  );
  assert.equal(toIsoDate(forecast.centerDate), "2026-08-13");
  assert.equal(forecast.typicalDays, 28);
  assert.equal(forecast.reliability.key, "personal");
  assert.ok(forecast.radius80 >= 3);
  assert.equal(forecast.expired, false);
});

test("variable history produces a wider, explicitly variable forecast", () => {
  const forecast = forecastCycle(
    [
      "2026-01-01",
      "2026-01-23",
      "2026-03-04",
      "2026-03-29",
      "2026-05-13",
      "2026-06-05",
      "2026-07-20",
    ],
    new Date("2026-07-24T12:00:00Z"),
  );
  assert.equal(forecast.reliability.key, "variable");
  assert.ok(forecast.radius80 >= 10);
  assert.ok(forecast.atypicalIntervals.length >= 3);
});

test("a forecast needs two recorded starts", () => {
  assert.equal(
    forecastCycle(["2026-07-01"], new Date("2026-07-24T12:00:00Z")),
    null,
  );
});

test("an elapsed prediction is marked expired rather than moved silently", () => {
  const forecast = forecastCycle(
    ["2026-01-01", "2026-01-29", "2026-02-26"],
    new Date("2026-05-01T12:00:00Z"),
  );
  assert.equal(forecast.expired, true);
  assert.equal(predictionSummary(forecast).expired, true);
});

test("backtesting appears only when prior history supports it", () => {
  const short = forecastCycle(
    ["2026-01-01", "2026-01-29", "2026-02-26"],
    new Date("2026-03-01T12:00:00Z"),
  );
  const long = forecastCycle(
    [
      "2026-01-01",
      "2026-01-29",
      "2026-02-27",
      "2026-03-27",
      "2026-04-25",
      "2026-05-23",
    ],
    new Date("2026-06-01T12:00:00Z"),
  );
  assert.equal(short.backtest, null);
  assert.ok(long.backtest.count >= 2);
  assert.ok(long.backtest.mae < 2);
});
