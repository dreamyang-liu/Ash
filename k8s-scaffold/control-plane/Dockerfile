FROM golang:1.24-alpine AS builder

WORKDIR /build
COPY ./* ./
RUN go mod download

RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags="-w -s" -o k8s-cp .

FROM alpine:3.19

RUN apk --no-cache add ca-certificates tzdata && \
    adduser -D -H -h /app appuser

WORKDIR /app
COPY --from=builder /build/k8s-cp .

USER appuser

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD wget -q --spider http://localhost:80/healthz || exit 1

ENTRYPOINT ["/app/k8s-cp"]
