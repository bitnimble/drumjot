import classNames from 'classnames';
import { PanelRightClose, PanelRightOpen } from 'lucide-react';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { SidebarStoreContext, SidebarPresenterContext } from './sidebar_contexts';
import { SIDEBAR_PANELS } from './panels/sidebar_panels';
import styles from './sidebar.module.css';

/**
 * Collapsible right-hand sidebar spanning the full page height. A persistent
 * vertical rail of icon buttons sits at the right edge; selecting an item opens
 * the panel area to its left. The rail's top holds the collapse/open button,
 * then a divider, then one button per registered panel (see
 * {@link SIDEBAR_PANELS}).
 *
 * The rail items and the panel body are both driven off that registry, so a
 * new panel is a single entry there, this component renders whichever the
 * registry lists.
 *
 * The sidebar is a flex sibling of the main content column, so opening it
 * narrows the score's measured width and the score's scroll virtualization
 * (driven by the container ResizeObserver) automatically stops rendering
 * anything that would fall under the panel.
 */
export const Sidebar = observer(function Sidebar() {
  const store = React.useContext(SidebarStoreContext);
  const presenter = React.useContext(SidebarPresenterContext);
  if (!store || !presenter) return null;
  const { expanded, activePanel } = store;
  const active = SIDEBAR_PANELS.find((p) => p.id === activePanel);

  return (
    <aside className={styles.sidebar} data-testid="sidebar" data-expanded={expanded || undefined}>
      {expanded && active && (
        <div className={styles.panel} data-testid="sidebar-panel">
          {active.render()}
        </div>
      )}
      <div className={styles.rail}>
        <button
          type="button"
          className={styles.railButton}
          onClick={() => presenter.toggleExpanded()}
          aria-label={expanded ? 'Collapse sidebar' : 'Open sidebar'}
          aria-expanded={expanded}
          title={expanded ? 'Collapse sidebar' : 'Open sidebar'}
          data-testid="sidebar-toggle"
        >
          {expanded ? <PanelRightClose size={18} /> : <PanelRightOpen size={18} />}
        </button>
        <div className={styles.railDivider} role="separator" aria-orientation="horizontal" />
        {SIDEBAR_PANELS.map((panel) => {
          const isActive = expanded && activePanel === panel.id;
          const Icon = panel.Icon;
          return (
            <button
              key={panel.id}
              type="button"
              className={classNames(styles.railButton, isActive && styles.railButtonActive)}
              onClick={() => presenter.selectPanel(panel.id)}
              aria-label={panel.label}
              aria-pressed={isActive}
              title={panel.label}
              data-testid={`sidebar-item-${panel.id}`}
            >
              <Icon size={18} />
            </button>
          );
        })}
      </div>
    </aside>
  );
});
