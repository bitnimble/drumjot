/**
 * Barrel for the laid-out jot model. The implementation is split into
 * three modules; importers keep using `from 'src/jot'`:
 *   - view_config.ts       the Pixels brand + ViewConfig
 *   - resolved_jot.ts      the Structural + Resolved layout types + the
 *                          RenderedJot layout engine + drum-offset pass
 *   - pattern_expansion.ts pattern/repeat expansion + element weight /
 *                          straightness / type-guard helpers
 */
export * from './view_config';
export * from './resolved_jot';
export * from './pattern_expansion';
