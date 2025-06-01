# Traffic Shadowing (Real Traffic Replay) - Design Doc & Roadmap

This document outlines a high-level design and roadmap for implementing traffic shadowing capabilities within this stress testing tool. The goal is to enable the replay of real production traffic patterns against a test environment or a new version of a service.

## 1. Goals

*   **Functional Testing:** Replay captured production traffic (preserving request timing and order as much as possible) to a new codebase or test cluster to verify functional correctness against real-world usage patterns.
*   **Non-Functional Testing:** Replay production traffic at varying throughput multipliers (e.g., 1x, 2x, 3x, 10x) to understand how the system scales and performs under increased load, helping to identify bottlenecks and capacity limits.
*   **Increased Confidence:** Enhance confidence in deployments by testing with realistic traffic scenarios that unit or integration tests might not cover.

## 2. Proposed High-Level Architecture/Flow

A traffic shadowing system typically involves the following components:

1.  **Log Source:**
    *   The origin of production traffic data.
    *   Examples: ElasticSearch, Kibana, server access logs (e.g., Nginx, Apache), cloud provider logging services (e.g., AWS CloudWatch Logs, Google Cloud Logging).

2.  **Puller/Extractor:**
    *   A component responsible for fetching or exporting logs from the chosen Log Source.
    *   This might involve API calls, database queries, or log file streaming.
    *   Output could be raw log files or an intermediate structured format.

3.  **Log Parser:**
    *   Transforms the raw or semi-structured logs into a standardized, replayable format.
    *   **Proposed Format:** JSON objects, where each object represents a single request and contains details like:
        *   Timestamp (original request time)
        *   HTTP Method (GET, POST, PUT, etc.)
        *   Request URL (path and query parameters)
        *   Request Headers
        *   Request Body (if any)
    *   Example: `{"timestamp": "2023-10-27T10:20:30.123Z", "method": "POST", "url": "/api/v1/items", "headers": {"Content-Type": "application/json", "X-User-ID": "123"}, "body": "{\"name\": \"widget\"}"}`
    *   This component could output a stream of JSON objects or a JSON file containing an array of these objects.

4.  **Data Sanitization/Manipulation (Rule Engine - Optional but often necessary):**
    *   Addresses security and environment-specific concerns before replaying traffic.
    *   **Key Functions:**
        *   **Filtering:** Remove requests that should not be replayed (e.g., admin actions, specific user data).
        *   **Anonymization/Obfuscation:** Mask or replace sensitive data in URLs, headers, or bodies (e.g., PII, API keys, tokens).
        *   **URL Rewriting:** Modify request URLs to target the test environment instead of production.
        *   **Header Manipulation:** Add, remove, or modify headers (e.g., inject test-specific headers, remove production authentication tokens).
        *   **Data Seeding:** Potentially pre-load the test environment with necessary data based on incoming requests.
    *   A "Rule Engine" could allow defining flexible rules for these transformations.

5.  **Simulator/Replayer:**
    *   The core component that takes the parsed (and possibly manipulated) requests and sends them to the target test system.
    *   **Key Features:**
        *   **Target Configuration:** Specify the base URL and other connection parameters for the test system.
        *   **Timing Control:**
            *   Replay with original request timing (maintaining inter-request delays).
            *   Replay as fast as possible (for load testing).
            *   Replay at a defined rate (requests per second).
        *   **Throughput Multiplication:** Replay traffic at N times the original rate (e.g., 2x, 5x). This might involve adjusting inter-request delays or increasing concurrency.
        *   **Concurrency Control:** Manage the number of concurrent requests being sent to the target.

6.  **Logger/Comparer (For Analysis & Future Enhancements):**
    *   **Logging:** Record the outcome of each replayed request (e.g., status code, latency, response body/headers if needed).
    *   **Comparison ("Tap Compare" - Future):**
        *   A more advanced feature where traffic is sent to both an old (baseline) version and a new (candidate) version of the service simultaneously.
        *   Responses from both versions are captured and compared to identify regressions or unexpected differences. Tools like Twitter's "Diffy" pioneered this concept.

## 3. Challenges

Implementing a robust traffic shadowing system involves several challenges:

*   **Security & Compliance:** Handling production data (which may contain sensitive customer information) requires strict security measures, data anonymization, and compliance with data privacy regulations.
*   **Data Mutability:** If the service under test modifies data, replaying traffic can alter the state of the test environment. Strategies are needed to isolate test runs or reset the environment.
*   **Environment Parity & Setup:** Creating a test environment that accurately mirrors production (including data, configurations, and dependencies) can be complex and costly.
*   **External Dependencies:** Services often interact with other downstream services. The shadowing setup must decide how to handle these external calls:
    *   Mock or stub external dependencies.
    *   Allow calls to test instances of dependencies.
    *   Carefully manage calls to live production dependencies (generally avoided).
*   **Resource Intensive:** Capturing, storing, and replaying large volumes of traffic can be resource-intensive.
*   **Complexity:** Building and maintaining the shadowing pipeline itself adds operational overhead.

## 4. Roadmap / TODOs (High-Level Phased Approach)

This roadmap outlines a potential phased approach to developing traffic shadowing capabilities.

**Phase 1: Foundational Local Replay**
*   [ ] Define a standard JSON format for replayable requests.
*   [ ] Develop a basic Log Parser capable of converting sample log files (e.g., simple Nginx access logs, or manually created JSON) into the standard replay format. Output to a local JSON file.
*   [ ] Develop a basic Simulator/Replayer that can:
    *   Read requests from the local JSON file.
    *   Replay them sequentially to a configurable target URL.
    *   Support basic timing (e.g., replay with a fixed delay between requests).
*   [ ] Basic CLI options for specifying input file and target URL.

**Phase 2: Enhanced Replay & Initial Log Source Integration**
*   [ ] Enhance Simulator/Replayer:
    *   Implement more sophisticated timing (e.g., replay based on original timestamps, configurable rate).
    *   Introduce basic concurrency for replaying requests.
*   [ ] Develop an initial Puller/Extractor for one specific Log Source (e.g., export from ElasticSearch via its API, assuming logs are already there).
*   [ ] Integrate Puller/Extractor output with the Log Parser.

**Phase 3: Throughput Multiplication & Basic Data Manipulation**
*   [ ] Enhance Simulator/Replayer:
    *   Implement throughput multiplication (e.g., replay at 2x, 3x the original rate). This will likely involve adjusting concurrency and/or inter-request delays.
*   [ ] Introduce basic Data Sanitization/Manipulation:
    *   Allow simple URL rewriting (e.g., change hostname).
    *   Allow adding/modifying a fixed set of headers.

**Phase 4: Advanced Data Manipulation & Rule Engine**
*   [ ] Design and implement a more flexible Rule Engine for data sanitization and manipulation (addressing filtering, anonymization, complex transformations).
*   [ ] Enhance security around handling potentially sensitive data.

**Phase 5: Reporting, Analysis & "Tap Compare" (Stretch Goal)**
*   [ ] Implement comprehensive logging of replayed request outcomes.
*   [ ] Develop basic reporting/analysis of replay results (e.g., success/error rates, latency statistics).
*   [ ] Investigate and potentially prototype a "Tap Compare" functionality for comparing responses from two service versions.

This roadmap provides a structured approach to incrementally building a powerful traffic shadowing feature. Each phase should include thorough testing and documentation.
