# Versionado Semántico - Usuarios Sintéticos

## Formato de Versión

```
vMAJOR.MINOR.PATCH
```

| Componente | Cuándo incrementar |
|------------|-------------------|
| **MAJOR** | Cambios incompatibles, rediseño completo, migración de datos |
| **MINOR** | Nueva funcionalidad compatible (nuevo módulo, feature, integración) |
| **PATCH** | Correcciones de bugs, ajustes de UI, fixes de seguridad |

## Convención de Tags

### Formato del Tag
```bash
git tag -a vX.Y.Z -m "Mensaje descriptivo"
```

### Estructura del Mensaje
```
[TIPO] Descripción breve

Cambios:
- Cambio 1
- Cambio 2

Estado: FUNCIONAL | EN DESARROLLO
Restaurar: git checkout vX.Y.Z
```

### Tipos de Release
- `[RELEASE]` - Versión estable, lista para producción
- `[FEATURE]` - Nueva funcionalidad agregada
- `[HOTFIX]` - Corrección urgente de bug
- `[SECURITY]` - Parche de seguridad

## Comandos Útiles

### Ver todas las versiones
```bash
git tag -l -n1
```

### Ver detalle de una versión
```bash
git show v1.0.0
```

### Restaurar a última versión funcional
```bash
# Ver versiones disponibles
git tag -l

# Restaurar (crear branch desde tag)
git checkout -b restore-vX.Y.Z vX.Y.Z

# O restaurar directo (detached HEAD)
git checkout vX.Y.Z
```

### Comparar versiones
```bash
git diff v1.0.0..v1.1.0
```

### Ver historial entre versiones
```bash
git log v1.0.0..v1.1.0 --oneline
```

## Flujo de Trabajo

```
Desarrollo
    │
    ▼
Commit con mensaje descriptivo
    │
    ▼
¿Funcionalidad completa?
    │
   Sí ──► git tag -a vX.Y.Z -m "descripción"
    │
    ▼
git push origin main --tags
```

## Historial de Versiones

| Versión | Fecha | Tipo | Descripción |
|---------|-------|------|-------------|
| v0.1.0 | 2026-01-21 | FEATURE | Setup inicial - Backend FastAPI + estructura |

---

**Regla de Oro:** Solo crear tags en commits que dejen el proyecto en estado funcional.
