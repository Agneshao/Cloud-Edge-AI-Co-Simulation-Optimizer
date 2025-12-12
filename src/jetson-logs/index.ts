import { ActorState } from "@liquidmetal-ai/raindrop-framework";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { Env } from './raindrop.gen.js';

export const implementation = {
  name: "jetson-logs",
  version: "1.0.0",
}

// Shared Cerebras prediction schema - used by both predict_latency and evaluate_config
interface CerebrasPrediction {
  predicted_latency_ms: number;
  predicted_temp_c: number;
  predicted_power_w: number;
  risk_level: "low" | "medium" | "high";
  recommended_safe_mode: "fast" | "safe";
  explanation: string;
}

const CEREBRAS_SYSTEM_PROMPT = `You are an expert NVIDIA Jetson edge AI performance analyst. Given current device metrics and a proposed configuration change, predict the expected performance.

You understand these relationships:
- FP32 -> FP16: ~1.5-2x speedup, minimal accuracy loss
- FP16 -> INT8: ~1.3-1.8x speedup, potential accuracy loss (1-3%)
- Batch size increase: improves throughput but increases latency per frame
- Resolution reduction: quadratic latency reduction (640->416 = ~1.6x speedup)
- Power mode affects clock speeds: MAXN > 30W > 15W > 10W
- Higher temps lead to thermal throttling above 80C

CRITICAL: You MUST respond with ONLY valid JSON. No markdown, no code blocks, no extra text.
Respond with exactly this JSON structure:
{
  "predicted_latency_ms": <number>,
  "predicted_temp_c": <number>,
  "predicted_power_w": <number>,
  "risk_level": "<low|medium|high>",
  "recommended_safe_mode": "<fast|safe>",
  "explanation": "<brief reasoning>"
}

Rules for risk_level:
- "low": safe config, no concerns
- "medium": some tradeoffs, monitor recommended
- "high": potential issues with thermal, stability, or accuracy

Rules for recommended_safe_mode:
- "fast": prioritize speed, acceptable for stable configs
- "safe": prioritize stability, use when risk_level is medium or high`;

// Helper: Parse Cerebras JSON response with strict schema
function parseCerebrasResponse(aiResponse: string): { prediction: CerebrasPrediction | null; error?: string; raw?: string } {
  if (!aiResponse || aiResponse.trim() === "") {
    return { prediction: null, error: "Empty response from AI" };
  }

  let parsed: any;
  try {
    // Try direct parse first
    parsed = JSON.parse(aiResponse.trim());
  } catch {
    // Try extracting JSON from markdown code blocks or extra text
    const jsonMatch = aiResponse.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      return { prediction: null, error: "No JSON object found in response", raw: aiResponse };
    }
    try {
      parsed = JSON.parse(jsonMatch[0]);
    } catch (e) {
      return { prediction: null, error: `JSON parse error: ${e instanceof Error ? e.message : String(e)}`, raw: aiResponse };
    }
  }

  // Validate required fields
  const requiredFields = ["predicted_latency_ms", "predicted_temp_c", "predicted_power_w", "risk_level", "recommended_safe_mode", "explanation"];
  const missingFields = requiredFields.filter(f => !(f in parsed));
  if (missingFields.length > 0) {
    return { prediction: null, error: `Missing required fields: ${missingFields.join(", ")}`, raw: aiResponse };
  }

  // Validate types and values
  if (typeof parsed.predicted_latency_ms !== "number") {
    parsed.predicted_latency_ms = parseFloat(parsed.predicted_latency_ms) || 0;
  }
  if (typeof parsed.predicted_temp_c !== "number") {
    parsed.predicted_temp_c = parseFloat(parsed.predicted_temp_c) || 0;
  }
  if (typeof parsed.predicted_power_w !== "number") {
    parsed.predicted_power_w = parseFloat(parsed.predicted_power_w) || 0;
  }
  if (!["low", "medium", "high"].includes(parsed.risk_level)) {
    parsed.risk_level = "medium";
  }
  if (!["fast", "safe"].includes(parsed.recommended_safe_mode)) {
    parsed.recommended_safe_mode = "safe";
  }
  if (typeof parsed.explanation !== "string") {
    parsed.explanation = String(parsed.explanation || "");
  }

  return {
    prediction: {
      predicted_latency_ms: Math.round(parsed.predicted_latency_ms * 100) / 100,
      predicted_temp_c: Math.round(parsed.predicted_temp_c * 100) / 100,
      predicted_power_w: Math.round(parsed.predicted_power_w * 100) / 100,
      risk_level: parsed.risk_level,
      recommended_safe_mode: parsed.recommended_safe_mode,
      explanation: parsed.explanation,
    }
  };
}

// Helper: Parse a CSV line handling quoted fields with commas
function parseCSVLine(line: string): string[] {
  const result: string[] = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];

    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === ',' && !inQuotes) {
      result.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }
  result.push(current.trim());
  return result;
}

// Helper: Parse CSV content into array of objects
function parseCSV(csvContent: string): any[] {
  const lines = csvContent.trim().split('\n');
  if (lines.length < 2) return [];

  const headerLine = lines[0];
  if (!headerLine) return [];

  const headers = parseCSVLine(headerLine);
  const data: any[] = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    if (!line) continue;

    const values = parseCSVLine(line);
    // Allow some flexibility - skip only if way off
    if (values.length < headers.length - 2 || values.length > headers.length + 2) continue;

    const row: any = {};
    headers.forEach((header, idx) => {
      const val = values[idx]?.trim() || '';
      // Try to parse as number
      const num = parseFloat(val);
      row[header] = isNaN(num) ? val : num;
    });
    data.push(row);
  }
  return data;
}

// Helper: Load log data from SmartBucket
async function loadLogFromBucket(env: Env, filename: string): Promise<{ data: any[] | null; error?: string }> {
  try {
    const object = await env.LOGS.get(`logs/${filename}`);
    if (!object) {
      return { data: null, error: `Object not found: logs/${filename}` };
    }
    const csvContent = await object.text();
    if (!csvContent || csvContent.length === 0) {
      return { data: null, error: "File is empty" };
    }
    const parsed = parseCSV(csvContent);
    if (parsed.length === 0) {
      return { data: null, error: `CSV parsing returned 0 rows. Content length: ${csvContent.length}, first 200 chars: ${csvContent.substring(0, 200)}` };
    }
    return { data: parsed };
  } catch (err) {
    return { data: null, error: `Exception: ${err instanceof Error ? err.message : String(err)}` };
  }
}

// Helper: List all logs in SmartBucket
async function listLogsInBucket(env: Env): Promise<{ filename: string; size: number; uploaded: string }[]> {
  try {
    const listing = await env.LOGS.list({ prefix: 'logs/' });
    return listing.objects.map(obj => ({
      filename: obj.key.replace('logs/', ''),
      size: obj.size,
      uploaded: obj.uploaded?.toISOString() || 'unknown',
    }));
  } catch {
    return [];
  }
}

function computeStats(data: any[], filename: string) {
  // Use actual CSV column names from real Jetson logs
  const latencies = data.map(d => d.end_to_end_ms).filter(v => !isNaN(v));
  const trtLatencies = data.map(d => d.trt_latency_ms).filter(v => !isNaN(v));
  const temps = data.map(d => d.gpu_temp_C).filter(v => !isNaN(v));
  const powers = data.map(d => d.power_mW).filter(v => !isNaN(v));
  const gpuUtils = data.map(d => d.gpu_util_percent).filter(v => !isNaN(v));
  const queueDelays = data.map(d => d.queue_delay_ms).filter(v => !isNaN(v));

  const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;
  const percentile = (arr: number[], p: number): number => {
    const sorted = [...arr].sort((a, b) => a - b);
    const idx = Math.floor(sorted.length * p);
    return sorted[idx] ?? 0;
  };
  const std = (arr: number[]) => {
    const mean = avg(arr);
    return Math.sqrt(arr.reduce((sum, x) => sum + (x - mean) ** 2, 0) / arr.length);
  };

  const maxDrops = Math.max(...data.map(d => d.frame_drop));
  const avgLatency = avg(latencies);

  return {
    file: filename,
    total_frames: data.length,
    latency: {
      avg_latency_ms: Math.round(avgLatency * 1000) / 1000,
      p50_latency_ms: Math.round(percentile(latencies, 0.5) * 1000) / 1000,
      p95_latency_ms: Math.round(percentile(latencies, 0.95) * 1000) / 1000,
      p99_latency_ms: Math.round(percentile(latencies, 0.99) * 1000) / 1000,
      max_latency_ms: Math.round(Math.max(...latencies) * 1000) / 1000,
      min_latency_ms: Math.round(Math.min(...latencies) * 1000) / 1000,
      jitter_std_ms: Math.round(std(latencies) * 1000) / 1000,
      avg_trt_inference_ms: Math.round(avg(trtLatencies) * 1000) / 1000,
    },
    throughput: {
      estimated_fps: Math.round((1000 / avgLatency) * 100) / 100,
    },
    reliability: {
      total_frame_drops: maxDrops,
      drop_rate: Math.round((maxDrops / data.length) * 10000) / 10000,
      avg_queue_delay_ms: Math.round(avg(queueDelays) * 1000) / 1000,
      max_queue_delay_ms: Math.round(Math.max(...queueDelays) * 1000) / 1000,
    },
    thermal: {
      avg_temp_c: Math.round(avg(temps) * 100) / 100,
      max_temp_c: Math.round(Math.max(...temps) * 100) / 100,
      min_temp_c: Math.round(Math.min(...temps) * 100) / 100,
    },
    power: {
      avg_power_mw: Math.round(avg(powers) * 100) / 100,
      max_power_mw: Math.round(Math.max(...powers) * 100) / 100,
      avg_power_w: Math.round(avg(powers) / 10) / 100,
    },
    gpu: {
      avg_util_percent: Math.round(avg(gpuUtils) * 100) / 100,
      max_util_percent: Math.round(Math.max(...gpuUtils) * 100) / 100,
    },
    deployment_config: {
      engine_name: data[0]?.engine_name || "unknown",
      engine_precision: data[0]?.engine_precision || "unknown",
      engine_batch: data[0]?.engine_batch || 1,
      engine_shape: data[0]?.engine_shape || "unknown",
      jetson_mode: data[0]?.jetson_mode || "unknown",
      platform: data[0]?.platform || "unknown",
      tensorrt_version: data[0]?.tensorrt_version || "unknown",
      cuda_version: data[0]?.cuda_version || "unknown",
    },
  };
}

export default (server: McpServer, env: Env, state: ActorState) => {
  // Tool: list_logs - List all available Jetson inference log files from SmartBucket
  server.registerTool("list_logs",
    {
      title: "List Logs",
      description: "List all available Jetson inference log files stored in SmartBucket",
      inputSchema: {},
    },
    async (_args: {}, { sendNotification }) => {
      await sendNotification({
        method: "notifications/message",
        params: {
          level: "info",
          data: "Fetching available log files from SmartBucket...",
        },
      });

      const files = await listLogsInBucket(env);

      const result = {
        file_count: files.length,
        storage: "SmartBucket",
        files,
      };

      return {
        content: [{
          type: "text",
          text: JSON.stringify(result, null, 2)
        }]
      };
    });

  // Tool: load_log - Load a specific log from SmartBucket and return performance statistics
  server.registerTool("load_log",
    {
      title: "Load Log",
      description: "Load a specific Jetson inference log from SmartBucket and return performance statistics including latency, drop rate, temperature, and deployment configuration",
      inputSchema: {
        filename: z.string().describe("Name of the CSV log file to load (e.g., 'inference_log_20251201_211136.csv')"),
      },
    },
    async ({ filename }: { filename: string }, { sendNotification }) => {
      await sendNotification({
        method: "notifications/message",
        params: {
          level: "info",
          data: `Loading ${filename} from SmartBucket...`,
        },
      });

      const result = await loadLogFromBucket(env, filename);

      if (!result.data || result.data.length === 0) {
        const availableFiles = await listLogsInBucket(env);
        return {
          content: [{
            type: "text",
            text: JSON.stringify({
              error: `Failed to load file: ${filename}`,
              debug: result.error,
              available_files: availableFiles.map(f => f.filename),
              hint: "Use upload_log tool to upload CSV files to SmartBucket first",
            }, null, 2)
          }]
        };
      }

      const stats = computeStats(result.data, filename);

      return {
        content: [{
          type: "text",
          text: JSON.stringify(stats, null, 2)
        }]
      };
    });

  // Tool: recommend_config - Get configuration recommendations
  server.registerTool("recommend_config",
    {
      title: "Recommend Configuration",
      description: "Analyze log statistics from SmartBucket and provide deployment configuration recommendations for optimizing Jetson inference performance",
      inputSchema: {
        filename: z.string().describe("Name of the log file to analyze for recommendations"),
      },
    },
    async ({ filename }: { filename: string }, { sendNotification }) => {
      await sendNotification({
        method: "notifications/message",
        params: {
          level: "info",
          data: `Loading ${filename} from SmartBucket for analysis...`,
        },
      });

      const result = await loadLogFromBucket(env, filename);

      if (!result.data || result.data.length === 0) {
        const availableFiles = await listLogsInBucket(env);
        return {
          content: [{
            type: "text",
            text: JSON.stringify({
              error: `Failed to load file: ${filename}`,
              debug: result.error,
              available_files: availableFiles.map(f => f.filename),
            }, null, 2)
          }]
        };
      }

      const stats = computeStats(result.data, filename);

      const recommendations: string[] = [];

      // Thermal analysis
      if (stats.thermal.max_temp_c > 75) {
        recommendations.push("HIGH TEMPERATURE: Max GPU temp exceeds 75C. Consider switching to a lower power mode (10W) or improving cooling.");
      } else if (stats.thermal.avg_temp_c > 60) {
        recommendations.push("MODERATE TEMPERATURE: Average temp is elevated. Monitor for thermal throttling during extended runs.");
      }

      // Latency analysis
      if (stats.latency.p95_latency_ms > 50) {
        recommendations.push("HIGH LATENCY: P95 latency exceeds 50ms. Consider using INT8 precision or reducing input resolution to 416x416.");
      }
      if (stats.latency.jitter_std_ms > 5) {
        recommendations.push("HIGH JITTER: Latency variance is high. Consider pinning CPU/GPU frequencies or using exclusive mode.");
      }

      // Drop rate analysis
      if (stats.reliability.drop_rate > 0.05) {
        recommendations.push("HIGH DROP RATE: Frame drops exceed 5%. Reduce input FPS or use a lighter model (YOLOv8n instead of YOLOv8s).");
      }

      // Queue delay analysis
      if (stats.reliability.avg_queue_delay_ms > 40) {
        recommendations.push("HIGH QUEUE DELAY: Processing backpressure detected. Consider increasing batch size or adding frame skipping logic.");
      }

      // Power analysis
      if (stats.power.avg_power_w > 12) {
        recommendations.push("HIGH POWER: Running near power limit. For battery applications, switch to 15W or 10W mode.");
      }

      // GPU utilization
      if (stats.gpu.avg_util_percent < 50) {
        recommendations.push("LOW GPU UTILIZATION: GPU is underutilized. Consider increasing batch size to 2 or 4 for better throughput.");
      }

      if (recommendations.length === 0) {
        recommendations.push("OPTIMAL: Current configuration appears well-tuned for the workload.");
      }

      const response = {
        filename,
        current_config: stats.deployment_config,
        metrics_summary: {
          avg_latency_ms: stats.latency.avg_latency_ms,
          p95_latency_ms: stats.latency.p95_latency_ms,
          estimated_fps: stats.throughput.estimated_fps,
          drop_rate: stats.reliability.drop_rate,
          avg_temp_c: stats.thermal.avg_temp_c,
          avg_power_w: stats.power.avg_power_w,
        },
        recommendations,
      };

      return {
        content: [{
          type: "text",
          text: JSON.stringify(response, null, 2)
        }]
      };
    });

  // Tool: predict_latency - Use Cerebras SmartInference to predict latency for new configs
  server.registerTool("predict_latency",
    {
      title: "Predict Latency",
      description: "Use Cerebras AI to predict inference performance for a proposed deployment configuration. Returns standardized prediction with latency, temp, power, risk level, and recommended mode.",
      inputSchema: {
        current_metrics: z.object({
          avg_latency_ms: z.number().describe("Current average latency in milliseconds"),
          avg_temp_c: z.number().describe("Current average GPU temperature in Celsius"),
          gpu_util_percent: z.number().describe("Current GPU utilization percentage"),
          power_w: z.number().describe("Current power consumption in watts"),
        }).describe("Current performance metrics from the device"),
        proposed_config: z.object({
          precision: z.enum(["FP32", "FP16", "INT8"]).describe("Model precision (FP32, FP16, or INT8)"),
          batch_size: z.number().describe("Batch size (1, 2, 4, 8)"),
          resolution: z.number().describe("Input resolution (320, 416, 512, 640)"),
          power_mode: z.enum(["10W", "15W", "30W", "MAXN"]).describe("Jetson power mode"),
        }).describe("Proposed deployment configuration to predict"),
      },
    },
    async ({ current_metrics, proposed_config }: {
      current_metrics: {
        avg_latency_ms: number;
        avg_temp_c: number;
        gpu_util_percent: number;
        power_w: number;
      };
      proposed_config: {
        precision: string;
        batch_size: number;
        resolution: number;
        power_mode: string;
      };
    }, { sendNotification }) => {
      await sendNotification({
        method: "notifications/message",
        params: {
          level: "info",
          data: "Calling Cerebras SmartInference for latency prediction...",
        },
      });

      const userPrompt = `Current device metrics:
- Average latency: ${current_metrics.avg_latency_ms}ms
- GPU temperature: ${current_metrics.avg_temp_c}°C
- GPU utilization: ${current_metrics.gpu_util_percent}%
- Power consumption: ${current_metrics.power_w}W

Proposed configuration:
- Precision: ${proposed_config.precision}
- Batch size: ${proposed_config.batch_size}
- Input resolution: ${proposed_config.resolution}x${proposed_config.resolution}
- Power mode: ${proposed_config.power_mode}

Predict the expected performance for this configuration change.`;

      try {
        const response = await env.AI.run('llama-3.3-70b', {
          model: 'llama-3.3-70b',
          messages: [
            { role: "system", content: CEREBRAS_SYSTEM_PROMPT },
            { role: "user", content: userPrompt }
          ],
          max_tokens: 500,
          temperature: 0.2,
        });

        const aiResponse = response.choices[0]?.message?.content || "";
        const parsed = parseCerebrasResponse(aiResponse);

        if (parsed.prediction) {
          // Return the prediction directly as the standardized schema
          return {
            content: [{
              type: "text",
              text: JSON.stringify(parsed.prediction, null, 2)
            }]
          };
        } else {
          // Return error with debug info
          return {
            content: [{
              type: "text",
              text: JSON.stringify({
                error: "Failed to parse AI prediction",
                details: parsed.error,
                raw_response: parsed.raw,
              }, null, 2)
            }]
          };
        }
      } catch (error) {
        return {
          content: [{
            type: "text",
            text: JSON.stringify({
              error: "Failed to call SmartInference",
              details: error instanceof Error ? error.message : String(error),
            }, null, 2)
          }]
        };
      }
    });

  // Tool: evaluate_config - Full safety and performance evaluation for UI-supplied configs
  server.registerTool("evaluate_config",
    {
      title: "Evaluate Configuration",
      description: "Comprehensive safety and performance evaluation for a proposed deployment configuration. Uses AI prediction to assess if the config is safe to deploy. Returns standardized prediction schema.",
      inputSchema: {
        log_filename: z.string().describe("Name of the log file to use as baseline metrics"),
        proposed_config: z.object({
          precision: z.enum(["FP32", "FP16", "INT8"]).describe("Model precision"),
          batch_size: z.number().describe("Batch size (1, 2, 4, 8)"),
          resolution: z.number().describe("Input resolution (320, 416, 512, 640)"),
          power_mode: z.enum(["10W", "15W", "30W", "MAXN"]).describe("Jetson power mode"),
        }).describe("Proposed deployment configuration"),
        constraints: z.object({
          max_latency_ms: z.number().optional().describe("Maximum acceptable latency in ms"),
          max_temp_c: z.number().optional().describe("Maximum acceptable temperature in Celsius"),
          min_fps: z.number().optional().describe("Minimum required FPS"),
          max_power_w: z.number().optional().describe("Maximum power budget in watts"),
        }).optional().describe("Optional performance constraints"),
      },
    },
    async ({ log_filename, proposed_config, constraints }: {
      log_filename: string;
      proposed_config: {
        precision: string;
        batch_size: number;
        resolution: number;
        power_mode: string;
      };
      constraints?: {
        max_latency_ms?: number;
        max_temp_c?: number;
        min_fps?: number;
        max_power_w?: number;
      };
    }, { sendNotification }) => {
      await sendNotification({
        method: "notifications/message",
        params: {
          level: "info",
          data: `Loading ${log_filename} from SmartBucket for evaluation...`,
        },
      });

      // Load baseline metrics from SmartBucket
      const loadResult = await loadLogFromBucket(env, log_filename);

      if (!loadResult.data || loadResult.data.length === 0) {
        const availableFiles = await listLogsInBucket(env);
        return {
          content: [{
            type: "text",
            text: JSON.stringify({
              error: `Failed to load file: ${log_filename}`,
              debug: loadResult.error,
              available_files: availableFiles.map(f => f.filename),
            }, null, 2)
          }]
        };
      }

      const stats = computeStats(loadResult.data, log_filename);

      const current_metrics = {
        avg_latency_ms: stats.latency.avg_latency_ms,
        avg_temp_c: stats.thermal.avg_temp_c,
        gpu_util_percent: stats.gpu.avg_util_percent,
        power_w: stats.power.avg_power_w,
      };

      await sendNotification({
        method: "notifications/message",
        params: {
          level: "info",
          data: "Calling Cerebras AI for performance prediction...",
        },
      });

      const userPrompt = `Current baseline metrics:
- Average latency: ${current_metrics.avg_latency_ms}ms
- GPU temperature: ${current_metrics.avg_temp_c}°C
- GPU utilization: ${current_metrics.gpu_util_percent}%
- Power: ${current_metrics.power_w}W

Current config: ${stats.deployment_config.engine_precision}, batch ${stats.deployment_config.engine_batch}, ${stats.deployment_config.jetson_mode}

Proposed configuration:
- Precision: ${proposed_config.precision}
- Batch size: ${proposed_config.batch_size}
- Resolution: ${proposed_config.resolution}x${proposed_config.resolution}
- Power mode: ${proposed_config.power_mode}

Predict the expected performance for this configuration change.`;

      try {
        const response = await env.AI.run('llama-3.3-70b', {
          model: 'llama-3.3-70b',
          messages: [
            { role: "system", content: CEREBRAS_SYSTEM_PROMPT },
            { role: "user", content: userPrompt }
          ],
          max_tokens: 500,
          temperature: 0.2,
        });

        const aiResponse = response.choices[0]?.message?.content || "";
        const parsed = parseCerebrasResponse(aiResponse);

        if (!parsed.prediction) {
          return {
            content: [{
              type: "text",
              text: JSON.stringify({
                error: "Failed to parse AI prediction",
                details: parsed.error,
                raw_response: parsed.raw,
              }, null, 2)
            }]
          };
        }

        const prediction = parsed.prediction;

        // Evaluate safety against constraints
        const defaultConstraints = {
          max_latency_ms: 100,
          max_temp_c: 80,
          max_power_w: 30,
        };
        const effectiveConstraints = { ...defaultConstraints, ...constraints };

        const violations: string[] = [];

        if (prediction.predicted_latency_ms > effectiveConstraints.max_latency_ms) {
          violations.push(`Latency ${prediction.predicted_latency_ms}ms exceeds limit ${effectiveConstraints.max_latency_ms}ms`);
        }
        if (prediction.predicted_temp_c > effectiveConstraints.max_temp_c) {
          violations.push(`Temperature ${prediction.predicted_temp_c}C exceeds limit ${effectiveConstraints.max_temp_c}C`);
        }
        if (prediction.predicted_power_w > effectiveConstraints.max_power_w) {
          violations.push(`Power ${prediction.predicted_power_w}W exceeds budget ${effectiveConstraints.max_power_w}W`);
        }

        const safe_to_deploy = violations.length === 0 && prediction.risk_level !== "high";

        // Return both the standardized prediction AND evaluation context
        return {
          content: [{
            type: "text",
            text: JSON.stringify({
              // Standardized Cerebras prediction (same schema as predict_latency)
              prediction: {
                predicted_latency_ms: prediction.predicted_latency_ms,
                predicted_temp_c: prediction.predicted_temp_c,
                predicted_power_w: prediction.predicted_power_w,
                risk_level: prediction.risk_level,
                recommended_safe_mode: prediction.recommended_safe_mode,
                explanation: prediction.explanation,
              },
              // Evaluation context
              evaluation: {
                safe_to_deploy,
                violations,
              },
              baseline_metrics: current_metrics,
              proposed_config,
              constraints_used: effectiveConstraints,
            }, null, 2)
          }]
        };
      } catch (error) {
        return {
          content: [{
            type: "text",
            text: JSON.stringify({
              error: "Failed to evaluate configuration",
              details: error instanceof Error ? error.message : String(error),
            }, null, 2)
          }]
        };
      }
    });

  // Tool: upload_log - Upload a CSV log file to SmartBucket
  server.registerTool("upload_log",
    {
      title: "Upload Log",
      description: "Upload a Jetson inference log CSV file to SmartBucket for persistent storage",
      inputSchema: {
        filename: z.string().describe("Name for the log file (e.g., 'inference_log_20251201_211136.csv')"),
        csv_content: z.string().describe("The full CSV content of the log file, including headers"),
      },
    },
    async ({ filename, csv_content }: { filename: string; csv_content: string }, { sendNotification }) => {
      await sendNotification({
        method: "notifications/message",
        params: {
          level: "info",
          data: `Uploading ${filename} to SmartBucket...`,
        },
      });

      try {
        // Validate CSV structure
        const lines = csv_content.trim().split('\n');
        if (lines.length < 2) {
          return {
            content: [{
              type: "text",
              text: JSON.stringify({
                error: "Invalid CSV: must have header row and at least one data row",
              }, null, 2)
            }]
          };
        }

        // Check for required columns
        const headerLine = lines[0];
        if (!headerLine) {
          return {
            content: [{
              type: "text",
              text: JSON.stringify({
                error: "Invalid CSV: missing header row",
              }, null, 2)
            }]
          };
        }

        const headers = headerLine.split(',').map(h => h.trim());
        const requiredColumns = ['trt_latency_ms', 'end_to_end_ms', 'gpu_temp_C', 'power_mW'];
        const missingColumns = requiredColumns.filter(c => !headers.includes(c));

        if (missingColumns.length > 0) {
          return {
            content: [{
              type: "text",
              text: JSON.stringify({
                error: "Invalid CSV: missing required columns",
                missing: missingColumns,
                required: requiredColumns,
                found_headers: headers,
              }, null, 2)
            }]
          };
        }

        // Upload to SmartBucket
        await env.LOGS.put(`logs/${filename}`, csv_content, {
          httpMetadata: {
            contentType: 'text/csv',
          },
        });

        const dataRows = lines.length - 1;

        return {
          content: [{
            type: "text",
            text: JSON.stringify({
              success: true,
              filename,
              path: `logs/${filename}`,
              size_bytes: csv_content.length,
              data_rows: dataRows,
              storage: "SmartBucket",
            }, null, 2)
          }]
        };
      } catch (error) {
        return {
          content: [{
            type: "text",
            text: JSON.stringify({
              error: "Failed to upload log to SmartBucket",
              details: error instanceof Error ? error.message : String(error),
            }, null, 2)
          }]
        };
      }
    });

  // Tool: delete_log - Delete a log file from SmartBucket
  server.registerTool("delete_log",
    {
      title: "Delete Log",
      description: "Delete a Jetson inference log file from SmartBucket",
      inputSchema: {
        filename: z.string().describe("Name of the log file to delete"),
      },
    },
    async ({ filename }: { filename: string }, { sendNotification }) => {
      await sendNotification({
        method: "notifications/message",
        params: {
          level: "info",
          data: `Deleting ${filename} from SmartBucket...`,
        },
      });

      try {
        await env.LOGS.delete(`logs/${filename}`);

        return {
          content: [{
            type: "text",
            text: JSON.stringify({
              success: true,
              deleted: filename,
            }, null, 2)
          }]
        };
      } catch (error) {
        return {
          content: [{
            type: "text",
            text: JSON.stringify({
              error: "Failed to delete log from SmartBucket",
              details: error instanceof Error ? error.message : String(error),
            }, null, 2)
          }]
        };
      }
    });
}
