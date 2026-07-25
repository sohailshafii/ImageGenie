import type { TrainingMetricPoint } from '../api/types';

// A minimal inline-SVG cost curve (B3): training loss, and validation loss where
// it was evaluated, over steps. Deliberately dependency-free — the app ships no
// charting library, and answering "did the loss go down?" needs only a line and
// an axis. Colors come from the app's theme tokens so it works in light and dark;
// the val line is dashed so the two series stay distinguishable without color.

const WIDTH = 720;
const HEIGHT = 280;
const PADDING = { top: 16, right: 16, bottom: 36, left: 48 };

interface LinePoint {
  step: number;
  loss: number;
}

function toPath(
  linePoints: LinePoint[],
  scaleX: (step: number) => number,
  scaleY: (loss: number) => number,
): string {
  return linePoints
    .map((point, index) => {
      const command = index === 0 ? 'M' : 'L';
      return `${command} ${scaleX(point.step).toFixed(1)} ${scaleY(point.loss).toFixed(1)}`;
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

  const trainPoints: LinePoint[] = points.map((point) => ({
    step: point.step,
    loss: point.loss,
  }));
  const valPoints: LinePoint[] = points
    .filter((point) => point.valLoss !== null)
    .map((point) => ({ step: point.step, loss: point.valLoss as number }));

  const axisBottom = HEIGHT - PADDING.bottom;

  return (
    <figure className="cost-curve">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label="Training loss over steps"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* y and x axes */}
        <line className="cc-axis" x1={PADDING.left} y1={PADDING.top} x2={PADDING.left} y2={axisBottom} />
        <line className="cc-axis" x1={PADDING.left} y1={axisBottom} x2={WIDTH - PADDING.right} y2={axisBottom} />
        {/* y ticks: highest and lowest loss */}
        <text className="cc-tick" x={PADDING.left - 6} y={scaleY(maxLoss)} textAnchor="end" dominantBaseline="middle">
          {maxLoss.toFixed(2)}
        </text>
        <text className="cc-tick" x={PADDING.left - 6} y={scaleY(minLoss)} textAnchor="end" dominantBaseline="middle">
          {minLoss.toFixed(2)}
        </text>
        {/* x ticks: first and last step */}
        <text className="cc-tick" x={PADDING.left} y={axisBottom + 18} textAnchor="middle">
          {minStep}
        </text>
        <text className="cc-tick" x={WIDTH - PADDING.right} y={axisBottom + 18} textAnchor="end">
          {maxStep}
        </text>
        {/* series */}
        <path className="cc-line cc-train" d={toPath(trainPoints, scaleX, scaleY)} fill="none" />
        {valPoints.length > 0 && (
          <path className="cc-line cc-val" d={toPath(valPoints, scaleX, scaleY)} fill="none" />
        )}
      </svg>
      <figcaption className="cc-legend">
        <span className="cc-key cc-train">train loss</span>
        {valPoints.length > 0 && <span className="cc-key cc-val">val loss</span>}
        <span className="cc-axis-label">loss vs. step</span>
      </figcaption>
    </figure>
  );
}
