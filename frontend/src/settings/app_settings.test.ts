import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import { AppSettingsStore } from './app_settings_store';
import { AppSettingsPresenter } from './app_settings_presenter';

const STORAGE_KEY = 'drumjot.settings';

// bun's test runtime has no `localStorage`; stub an in-memory one (the
// presenter persists through the global, and its own guards no-op without it).
class MemoryStorage {
  private readonly m = new Map<string, string>();
  getItem(key: string): string | null {
    return this.m.has(key) ? (this.m.get(key) as string) : null;
  }
  setItem(key: string, value: string): void {
    this.m.set(key, value);
  }
  removeItem(key: string): void {
    this.m.delete(key);
  }
}

const globalRef = globalThis as { localStorage?: Storage };
const original = globalRef.localStorage;

beforeEach(() => {
  globalRef.localStorage = new MemoryStorage() as unknown as Storage;
});

afterEach(() => {
  globalRef.localStorage = original;
});

describe('AppSettingsStore.apiBase', () => {
  test('empty origin is origin-relative /api', () => {
    const store = new AppSettingsStore();
    store.transcriberUrl = '';
    expect(store.apiBase).toBe('/api');
  });

  test('absolute origin gets /api appended', () => {
    const store = new AppSettingsStore();
    store.transcriberUrl = 'https://drumjot.kumo.dev';
    expect(store.apiBase).toBe('https://drumjot.kumo.dev/api');
  });

  test('a trailing slash + surrounding whitespace are stripped', () => {
    const store = new AppSettingsStore();
    store.transcriberUrl = '  https://lan.box:8000/  ';
    expect(store.apiBase).toBe('https://lan.box:8000/api');
  });
});

describe('AppSettingsPresenter persistence', () => {
  test('writes changes to localStorage', () => {
    const presenter = new AppSettingsPresenter(new AppSettingsStore());
    presenter.setTranscriberUrl('https://example.test');
    presenter.setBackendMode('hosted');
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}');
    expect(saved).toEqual({ backendMode: 'hosted', transcriberUrl: 'https://example.test' });
  });

  test('a fresh presenter loads the saved values on boot', () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ backendMode: 'hosted', transcriberUrl: 'https://saved.test' }),
    );
    const store = new AppSettingsStore();
    new AppSettingsPresenter(store);
    expect(store.backendMode).toBe('hosted');
    expect(store.transcriberUrl).toBe('https://saved.test');
  });

  test('corrupt JSON leaves the defaults intact', () => {
    localStorage.setItem(STORAGE_KEY, '{not json');
    const store = new AppSettingsStore();
    const before = store.transcriberUrl;
    new AppSettingsPresenter(store);
    expect(store.backendMode).toBe('local');
    expect(store.transcriberUrl).toBe(before);
  });

  test('an unknown backendMode is ignored', () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ backendMode: 'bogus' }));
    const store = new AppSettingsStore();
    new AppSettingsPresenter(store);
    expect(store.backendMode).toBe('local');
  });
});
