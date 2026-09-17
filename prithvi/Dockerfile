# Prithvi-EO-2.0 (Sen4Map crops) service. Reuses the TerraMind image, which
# already carries torch 2.10 cu128 + terratorch 1.1.1 + the geo stack.
# Build context is the repo root: docker compose --profile prithvi build prithvi
FROM agrisense-terramind:latest
WORKDIR /app
ENV AGRISENSE_MAX_PX=0 \
    HF_HUB_OFFLINE=1 \
    PRITHVI_CKPT=/models/prithvi_sen4map_crops.ckpt
COPY satellite.py prithvi/service.py prithvi/validate_worldcover.py /app/
EXPOSE 8601
CMD ["python3", "service.py"]
