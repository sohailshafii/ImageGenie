import type { TrainingMetricPoint } from '../api/types';

// A minimal inline-SVG cost curve (B3): training loss, validation loss where it
// was evaluated, and validation accuracy on its own axis. Deliberately
// dependency-free — the app ships no charting library, and answering "did the
// loss go down, and did it learn anything?" needs only lines and two axes.
// Colors come from the app's theme tokens so it works in light and dark; the
// series are also dashed/dotted so they stay distinguishable without color.

const WIDTH = 720;
const HEIGHT = 280;
const PADDING = { top: 16, right: 48, bottom: 36, left: 48 };

interface LinePoint {
  step: number;
  value: number;
}

function toPath(
  linePoints: LinePoint[],
  scaleX: (step: number) => number,
  scaleY: (value: number) => number,
): string {
  return linePoints
    .map((point, index) => {
      const command = index === 0 ? 'M' : 'L';
      return `${command} ${scaleX(point.step).toFixed(1)} ${scaleY(point.value).toFixed(1)}`;
    })
    .join(' ');
}

export function CostCurve({ points }: { points: TrainingMetricPoint[] }) {
  if (points.length === 0) {
    return <p className="page-lead">No loss points were logged for this run.</p>;
  }

  const plotWidth = WIDTH - PADDING.left - PADDING.right;
  const plotHeight = HEIGHT - PADDING.top - PADDING.bottom;

  const steps = points.map((point) => point.step);
  // Include val losses in the y-range so a higher val curve isn't clipped.
  const losses = points.flatMap((point) =>
    point.valLoss === null ? [point.loss] : [point.loss, point.valLoss],
  );
  const minStep = Math.min(...steps);
  const maxStep = Math.max(...steps);
  const minLoss = Math.min(...losses);
  const maxLoss = Math.max(...losses);

  // Guard against a zero span (a single point, or a flat loss) dividing by zero.
  const stepSpan = maxStep - minStep || 1;
  const lossSpan = maxLoss - minLoss || 1;

  const scaleX = (step: number) => PADDING.left + ((step - minStep) / stepSpan) * plotWidth;
  // SVG y grows downward, so a lower loss must map to a larger y — hence 1 - ratio.
  const scaleY = (loss: number) =>
    PADDING.top + (1 - (loss - minLoss) / lossSpan) * plotHeight;

  // Accuracy gets its own axis with a **fixed 0..1 domain**, not the loss axis:
  // the two quantities share no units, and auto-fitting accuracy to its own
  // min/max would make a model stuck at the majority-class rate look like it was
  // climbing steeply. A fixed domain also makes the height directly comparable
  // between runs.
  const scaleAccuracy = (accuracy: number) =>
    PADDING.top + (1 - accuracy) * plotHeight;

  const trainPoints: LinePoint[] = points.map((point) => ({
    step: point.step,
    value: point.loss,
  }));
  const valPoints: LinePoint[] = points
    .filter((point) => point.valLoss !== null)
    .map((point) => ({ step: point.step, value: point.valLoss as number }));
  const accuracyPoints: LinePoint[] = points
    .filter((point) => point.valAccuracy !== null)
    .map((point) => ({ step: point.step, value: point.valAccuracy as number }));

  const axisBottom = HEIGHT - PADDING.bottom;
  const axisRight = WIDTH - PADDING.right;

  return (
    <figure className="cost-curve">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label="Training and validation loss, and validation accuracy, over steps"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* y and x axes */}
        <line className="cc-axis" x1={PADDING.left} y1={PADDING.top} x2={PADDING.left} y2={axisBottom} />
        <line className="cc-axis" x1={PADDING.left} y1={axisBottom} x2={axisRight} y2={axisBottom} />
        {/* y ticks: highest and lowest loss */}
        <text className="cc-tick" x={PADDING.left - 6} y={scaleY(maxLoss)} textAnchor="end" dominantBaseline="middle">
          {maxLoss.toFixed(2)}
        </text>
        <text className="cc-tick" x={PADDING.left - 6} y={scaleY(minLoss)} textAnchor="end" dominantBaseline="middle">
          {minLoss.toFixed(2)}
        </text>
        {/* right axis: accuracy, fixed 0..100% */}
        {accuracyPoints.length > 0 && (
          <>
            <line className="cc-axis" x1={axisRight} y1={PADDING.top} x2={axisRight} y2={axisBottom} />
            <text className="cc-tick" x={axisRight + 6} y={scaleAccuracy(1)} dominantBaseline="middle">
              100%
            </text>
            <text className="cc-tick" x={axisRight + 6} y={scaleAccuracy(0)} dominantBaseline="middle">
              0%
            </text>
          </>
        )}
        {/* x ticks: first and last step */}
        <text className="cc-tick" x={PADDING.left} y={axisBottom + 18} textAnchor="middle">
          {minStep}
        </text>
        <text className="cc-tick" x={axisRight} y={axisBottom + 18} textAnchor="end">
          {maxStep}
        </text>
        {/* series */}
        <path className="cc-line cc-train" d={toPath(trainPoints, scaleX, scaleY)} fill="none" />
        {valPoints.length > 0 && (
          <path className="cc-line cc-val" d={toPath(valPoints, scaleX, scaleY)} fill="none" />
        )}
        {accuracyPoints.length > 0 && (
          <path
            className="cc-line cc-acc"
            d={toPath(accuracyPoints, scaleX, scaleAccuracy)}
            fill="none"
          />
        )}
      </svg>
      <figcaption className="cc-legend">
        <span className="cc-key cc-train">train loss</span>
        {valPoints.length > 0 && <span className="cc-key cc-val">val loss</span>}
        {accuracyPoints.length > 0 && <span className="cc-key cc-acc">val accuracy</span>}
        {/* The caption names only the axes actually drawn — a run with no
            accuracy has no right axis, so promising one is just wrong. */}
        <span className="cc-axis-label">
          {accuracyPoints.length > 0
            ? 'loss (left) · accuracy (right) vs. step'
            : 'loss vs. step'}
        </span>
      </figcaption>
    </figure>
  );
}
