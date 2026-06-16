import classNames from 'classnames';
import { Layers, PanelRightClose, PanelRightOpen } from 'lucide-react';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { SidebarStoreContext, SidebarPresenterContext } from './sidebar_contexts';
import { LayersPanel } from './panels/layers_panel';
import styles from './sidebar.module.css';

/**
 * Collapsible right-hand sidebar spanning the full page height. A persistent
 * vertical rail of icon buttons sits at the right edge; selecting an item opens
 * the panel area to its left. The rail's top holds the collapse/open button,
 * then a divider, then the panel items (today only Layers).
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

  return (
    <aside className={styles.sidebar} data-testid="sidebar" data-expanded={expanded || undefined}>
      {expanded && (
        <div className={styles.panel} data-testid="sidebar-panel">
          {activePanel === 'layers' && <LayersPanel />}
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
        <button
          type="button"
          className={classNames(
            styles.railButton,
            expanded && activePanel === 'layers' && styles.railButtonActive
          )}
          onClick={() => presenter.selectPanel('layers')}
          aria-label="Layers"
          aria-pressed={expanded && activePanel === 'layers'}
          title="Layers"
          data-testid="sidebar-item-layers"
        >
          <Layers size={18} />
        </button>
      </div>
    </aside>
  );
});
