# Decisiones de diseño

| # | Decisión | Justificación |
|---|----------|---------------|
| D1 | Solo USD, multi-moneda listo | `CURRENCIES` en `src/data_loader.py`; añadir `"ECU"` escala sin tocar nada más |
| D2 | Target `avg` diario, horizonte 7d | Es la variable de referencia del mercado informal; 7d útil y estable para la serie |
| D3 | Re-exploración manual | El fit completo es caro (CPU); explorar a voluntad evita drift automático no revisado |
| D4 | Frontend estático | Sin backend: GitHub Pages sirve JSONs, costo cero, auditable |
| D5 | Modelos commiteados directo | AutoGluon con modelos ligeros (sin Chronos) cabe en git; LFS añade complejidad |
| D6 | fit() completo una sola vez (notebook) | Producción solo reentrena la receta: reproducible y barato |
| D7 | Disclaimer público | Transparencia y responsabilidad |

## Limitaciones

- **Serie corta**: ~2 años (729 días). Suficiente para patrones semanales, no
  para estacionalidad anual ni ciclos largos.
- **Mercado informal con shocks estructurales**: cambios de política, apagones,
  eventos macro no modelados. El modelo no los anticipa.
- **Una sola fuente**: eltoque.com vía cubanomic. Sin redundancia de datos.
- **Sin variables exógenas**: solo el precio. Oferta de divisas, liquidez,
  turismo, etc. no se incluyen.
- **Métrica MASE vs baselines**: naive y media móvil son baselines débiles;
  superarlos es condición necesaria, no suficiente.

## Disclaimer

**No es consejo financiero.** Este proyecto es experimental y con fines
educativos. Las predicciones pueden ser incorrectas y no deben usarse para
decisiones de inversión, compra o venta de divisas. El autor no se hace
responsable de pérdidas derivadas del uso de esta información.
