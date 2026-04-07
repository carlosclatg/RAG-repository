import {
    LoggerProvider,
    BatchLogRecordProcessor,
    ConsoleLogRecordExporter,
} from '@opentelemetry/sdk-logs';
import { OTLPLogExporter } from '@opentelemetry/exporter-logs-otlp-http';
import { Resource } from '@opentelemetry/resources';
import {
    ATTR_SERVICE_NAME,
    ATTR_SERVICE_VERSION,
} from '@opentelemetry/semantic-conventions';
import { logs, SeverityNumber, Logger } from '@opentelemetry/api-logs';

const OTEL_ENDPOINT =
    process.env.OTEL_EXPORTER_OTLP_LOGS_ENDPOINT ??
    'http://localhost:4318/v1/logs';

const resource = new Resource({
    [ATTR_SERVICE_NAME]: 'rag-local',
    [ATTR_SERVICE_VERSION]: '1.0.0',
});

const otlpExporter = new OTLPLogExporter({ url: OTEL_ENDPOINT });

const loggerProvider = new LoggerProvider({ resource });
loggerProvider.addLogRecordProcessor(
    new BatchLogRecordProcessor(otlpExporter),
);
loggerProvider.addLogRecordProcessor(
    new BatchLogRecordProcessor(new ConsoleLogRecordExporter()),
);

logs.setGlobalLoggerProvider(loggerProvider);

const logger: Logger = logs.getLogger('rag-local', '1.0.0');

export function logInfo(
    message: string,
    attributes?: Record<string, string | number | boolean>,
): void {
    logger.emit({
        severityNumber: SeverityNumber.INFO,
        severityText: 'INFO',
        body: message,
        attributes,
    });
}

export function logWarn(
    message: string,
    attributes?: Record<string, string | number | boolean>,
): void {
    logger.emit({
        severityNumber: SeverityNumber.WARN,
        severityText: 'WARN',
        body: message,
        attributes,
    });
}

export function logError(
    message: string,
    attributes?: Record<string, string | number | boolean>,
): void {
    logger.emit({
        severityNumber: SeverityNumber.ERROR,
        severityText: 'ERROR',
        body: message,
        attributes,
    });
}

export async function shutdownLogger(): Promise<void> {
    await loggerProvider.shutdown();
}
