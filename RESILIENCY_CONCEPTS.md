# Resiliency Concepts

This document summarizes key concepts for building resilient systems, focusing on how clients should interact with services.

## 1. Timeouts

- **Definition**: The maximum time a client will wait for a response from a service after sending a request.
- **Why**: Timeouts are crucial for preventing client resources (threads, connections, memory) from being exhausted by requests that are taking too long or have failed. They ensure that a client can move on and not wait indefinitely.
- **Best Practices**:
    - Apply timeouts to any remote call or external dependency.
    - Differentiate between connection timeouts (waiting to establish a connection) and request/read timeouts (waiting for data after connection).
- **Challenges**:
    - Setting timeouts too high can lead to resource consumption if many requests are slow.
    - Setting timeouts too low can cause excessive retries for normally functioning requests, potentially leading to cascading failures.
- **Choosing Values**:
    - Base timeout values on observed latency metrics (e.g., p99 or p99.9 latency for a specific call). A p99.9 target means accepting that 0.1% of good requests might time out.
    - Consider the end-to-end latency budget for user-facing operations.
- **Pitfalls**:
    - Underestimating network latency, especially over WAN.
    - Setting tight latency bounds that don't account for occasional system variance, requiring padding.
    - Forgetting to cover all aspects of a call: DNS lookup, TCP handshake, TLS negotiation, sending the request, and receiving the response.

## 2. Retries

- **Definition**: The act of sending the same request again after an initial attempt fails or times out.
- **Why**: Retries help systems recover from transient (temporary) or partial failures, improving the success rate of operations.
- **"Selfish" Nature**: From a system perspective, retries are "selfish" because an individual client uses more server resources to achieve its own success, potentially at the expense of other clients if the system is under stress.
- **Problematic Aspect**: Uncontrolled retries can significantly worsen an overload situation on a struggling service by amplifying the request volume.
- **Solution**: Implement backoff strategies with retries.
- **Advanced Considerations**:
    - **Multi-layer Retries**: Retries at multiple layers of an application stack (e.g., client library, application code, service mesh) can amplify load unexpectedly. Be aware of and coordinate retry strategies across layers.
    - **Token Buckets**: For local retry limits, a token bucket algorithm can be used to cap the rate of retries originating from a client, preventing it from overwhelming the server.
    - **Idempotency**: Retrying requests to APIs that have side effects (e.g., creating or modifying data) is only safe if the operations are idempotent (i.e., making the same request multiple times has the same effect as making it once).

## 3. Backoff

- **Definition**: The practice of waiting for a period of time before attempting to retry a failed request.
- **Purpose**: Backoff strategies prevent clients from overwhelming an already stressed or failing service with rapid retry attempts. They give the downstream system time to recover.
- **Common Pattern**:
    - **Exponential Backoff**: The wait time between retries increases exponentially with each failed attempt (e.g., wait 1s, then 2s, then 4s, etc.).
- **Improvements**:
    - **Capped Exponential Backoff**: Use exponential backoff but set a maximum limit on the wait time to prevent excessively long delays.
    - **Limit Number of Retries**: Always define a maximum number of retry attempts to avoid indefinite retrying.

## 4. Jitter

- **Definition**: The practice of adding a small, random amount of time to backoff intervals or before making initial requests.
- **Why**: Jitter helps to prevent "thundering herd" scenarios where many clients, recovering from a failure or starting a scheduled task simultaneously, retry or initiate requests at exactly the same time. By spreading out these requests, jitter smooths out traffic spikes.
- **Benefits**:
    - Reduces correlated retries, leading to more stable system behavior.
    - Helps downstream services scale more gracefully by avoiding sudden bursts of traffic.
    - Can potentially reduce the amount of server capacity needed to handle peak loads caused by synchronized client behavior.
- **Implementation**:
    - When using jitter for scheduled or batch work, ensure the random number generation is consistent or repeatable if necessary for debugging or analysis purposes.
    - Full Jitter (random value between 0 and the current exponential backoff delay) is often more effective than Equal Jitter (fixed fraction of randomness).
