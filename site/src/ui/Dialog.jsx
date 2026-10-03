import React, {useEffect, useId, useRef, useState} from 'react';
import {createPortal} from 'react-dom';
import Button from './Button.jsx';
import './dialog.css';

/** A top-layer panel whose content scrolls without enlarging the page. */
export default function Dialog({label, title = label, className = '', children}) {
  const dialog = useRef(null);
  const body = useRef(null);
  const closing = useRef(null);
  const titleId = useId();
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const panel = dialog.current;
    return () => {closing.current?.cancel(); panel?.close();};
  }, []);

  function show() {
    closing.current?.cancel();
    closing.current = null;
    delete dialog.current.dataset.closing;
    dialog.current.showModal();
    body.current.scrollTop = 0;
    setOpen(true);
  }

  function close() {
    const panel = dialog.current;
    if (!panel?.open || closing.current) return;
    const finish = () => {panel.close(); setOpen(false); closing.current = null;};
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches || document.hidden) {
      finish();
      return;
    }
    panel.dataset.closing = '';
    const animation = panel.animate(
      [{opacity: 1, transform: 'translateY(0)'}, {opacity: 0, transform: 'translateY(8px)'}],
      {duration: 140, easing: 'ease-in', fill: 'forwards'},
    );
    closing.current = animation;
    animation.finished.then(() => {finish(); animation.cancel();}).catch(() => {});
  }

  return <>
    <Button className={className} aria-haspopup="dialog" aria-expanded={open} onClick={show}>{label}</Button>
    {createPortal(<dialog ref={dialog} className="ui-dialog" aria-labelledby={titleId}
      onCancel={event => {event.preventDefault(); close();}}
      onClick={event => {if (event.target === event.currentTarget) close();}}>
      <div className="ui-dialog-header">
        <h2 id={titleId}>{title}</h2>
        <Button autoFocus aria-label={`Close ${title}`} onClick={close}>Close <span aria-hidden="true">×</span></Button>
      </div>
      <div ref={body} className="ui-dialog-body">{children}</div>
    </dialog>, document.body)}
  </>;
}
