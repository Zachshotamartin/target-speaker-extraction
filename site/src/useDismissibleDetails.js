import {useEffect} from 'react';
import {setDetailsOpen} from './motion.js';

// Native details semantics with light dismissal for floating help/popover content.
export function useDismissibleDetails(ref, scope) {
  useEffect(() => {
    function dismiss(event) {
      const details = ref.current;
      if (!details?.open) return;
      // Clicking noninteractive popup text can focus an ancestor such as <main>.
      // Pointer dismissal already handles that gesture; focus dismissal is for keys.
      if (event.type === 'focusin' && !event.target.matches(':focus-visible')) return;
      const escape = event.type === 'keydown' && event.key === 'Escape';
      if (event.type === 'keydown' && !escape) return;
      if (!escape && details.contains(event.target)) return;
      if (escape) event.preventDefault();
      setDetailsOpen(details, false, scope);
      if (escape) details.querySelector('summary')?.focus({preventScroll: true});
    }
    document.addEventListener('pointerdown', dismiss);
    document.addEventListener('focusin', dismiss);
    document.addEventListener('keydown', dismiss);
    return () => {
      document.removeEventListener('pointerdown', dismiss);
      document.removeEventListener('focusin', dismiss);
      document.removeEventListener('keydown', dismiss);
    };
  }, [ref, scope]);
}
