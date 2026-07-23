import { useEffect, useRef, useState } from 'react';

// A destructive action that arms on the first click and fires on the second. A
// two-step inline confirm rather than window.confirm: no blocking native dialog,
// and it makes a mis-click on a dense grid harmless — the first press only arms
// it. It disarms on a timeout or on blur, but deliberately NOT on mouse-leave:
// the armed label is wider than the idle one, so the small cursor move between
// the two clicks can exit the button's idle bounds, and disarming there would
// drop the confirming click onto whatever sits behind it (on a card, the link to
// the detail page).
export function ConfirmButton({
  onConfirm,
  idleLabel,
  armedLabel,
  busy,
  className = 'btn-secondary',
  title,
}: {
  onConfirm: () => void;
  idleLabel: React.ReactNode;
  armedLabel: React.ReactNode;
  busy?: boolean;
  className?: string;
  title?: string;
}) {
  const [armed, setArmed] = useState(false);
  const timer = useRef<number | undefined>(undefined);

  // Auto-disarm, so a card left armed doesn't fire on a much-later stray click.
  useEffect(() => {
    if (!armed) return;
    timer.current = window.setTimeout(() => setArmed(false), 4000);
    return () => window.clearTimeout(timer.current);
  }, [armed]);

  return (
    <button
      type="button"
      className={`${className}${armed ? ' is-armed' : ''}`}
      disabled={busy}
      title={title}
      onClick={() => {
        if (armed) {
          setArmed(false);
          onConfirm();
        } else {
          setArmed(true);
        }
      }}
      onBlur={() => setArmed(false)}
    >
      {armed ? armedLabel : idleLabel}
    </button>
  );
}
