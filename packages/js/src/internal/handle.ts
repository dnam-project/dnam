/**
 * Native resource lifecycle tracking (`FinalizationRegistry` + `Symbol.dispose`)
 * (`ref[sdk.ts.api.disposable-scopes]`).
 *
 * @module
 */

const finalizer =
  typeof FinalizationRegistry !== "undefined"
    ? new FinalizationRegistry<() => void>((cleanup) => {
        try {
          cleanup();
        } catch {
          // Ignore cleanup errors during GC finalization.
        }
      })
    : undefined;

/**
 * Registers an object with the global `FinalizationRegistry` and returns an idempotent
 * release function for explicit `.dispose()` / `[Symbol.dispose]()` calls.
 */
export function registerDisposable(owner: object, cleanup: () => void): () => void {
  const token = {};
  finalizer?.register(owner, cleanup, token);
  let disposed = false;
  return () => {
    if (disposed) {
      return;
    }
    disposed = true;
    finalizer?.unregister(token);
    cleanup();
  };
}
