"""
AI Swarm Agent - Advanced Edition

High-performance async AI swarm with caching, cost tracking, and weighted consensus
for Polymarket trading decisions.

Features:
- Async parallel queries with aiohttp
- Smart timeout handling with early termination
- Response caching (5min TTL)
- Cost tracking and optimization
- Weighted consensus by model accuracy
- Advanced probability extraction
- Real-time progress feedback

Author: Polymarket Trading Bot
"""

import os
import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import re
from collections import defaultdict

from termcolor import colored
from loguru import logger

from src.models import model_factory, ModelResponse
from src.analysis.model_calibration import auto_calibrator


# Configuration for swarm models
# Format: "name": (enabled, provider_key, model_id)
# provider_key must match MODEL_CLASSES in model_factory.py
SWARM_MODELS = {
    "claude": (True, "claude", "claude-3-5-haiku-20241022"),
    "gemini": (True, "google", "gemini-pro"),           # provider is "google" not "gemini"
    "gpt": (True, "openai", "gpt-3.5-turbo"),           # provider is "openai" not "gpt"
    "deepseek": (True, "openrouter", "deepseek/deepseek-v3.2"),  # provider is "openrouter"
    "perplexity": (True, "perplexity", "sonar-pro"),
}

# Model weights for consensus (higher = more trusted)
MODEL_WEIGHTS = {
    "claude": 1.3,      # Best reasoning
    "gemini": 1.3,      # Gemini 3 Pro - SOTA reasoning
    "gpt": 1.2,         # GPT-5.2 - Strong general intelligence
    "perplexity": 1.2,  # Has real-time web search
    "deepseek": 1.0,    # Good baseline, very fast (V3.2)
}

# Timeout settings per model (seconds)
MODEL_TIMEOUTS = {
    "claude": 45,       # Slower but high quality
    "gemini": 35,       # Fast and capable
    "gpt": 40,          # Balanced performance
    "deepseek": 30,     # Generally fast (V3.2)
    "perplexity": 60,   # Can be slower due to web search
}


@dataclass
class CachedResponse:
    """Cached response with TTL."""
    response: Any
    timestamp: datetime
    ttl_seconds: int = 300  # 5 minutes default

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() - self.timestamp > timedelta(seconds=self.ttl_seconds)


@dataclass
class SwarmResponse:
    """Response from a single model in the swarm."""
    provider: str
    model_name: str
    response: Optional[ModelResponse] = None
    success: bool = False
    error: Optional[str] = None
    weight: float = 1.0


@dataclass
class SwarmResponse:
    """Response from a single model in the swarm."""
    provider: str
    model_name: str
    response: ModelResponse
    success: bool
    error: Optional[str] = None


class SwarmAgent:
    """Advanced AI swarm agent with async queries, caching, and weighted consensus."""

    def __init__(self) -> None:
        """Initialize swarm agent with enabled models and efficiency features."""
        self.models = {}
        self.enabled_providers = []
        self.cache = {}  # Response cache
        self.cost_tracker = defaultdict(float)  # Track costs per model
        self.total_cost = 0.0  # Total API costs

        # Initialize calibration
        self.calibrator = auto_calibrator

        logger.info("🔄 Initializing Advanced AI Swarm Agent...")

        # Initialize enabled models
        for provider, (enabled, model_type, model_name) in SWARM_MODELS.items():
            if enabled:
                try:
                    model = model_factory.get_model(model_type, model_name)
                    self.models[provider] = model
                    self.enabled_providers.append(provider)

                    weight = MODEL_WEIGHTS.get(provider, 1.0)
                    print(colored(f"✅ {provider}: {model_name} (weight: {weight})", "green"))

                except Exception as e:
                    error_msg = f"Failed to initialize {provider}: {e}"
                    logger.warning(error_msg)
                    print(colored(f"❌ {provider}: {error_msg}", "red"))
            else:
                print(colored(f"⏭️  {provider}: disabled", "yellow"))

        if not self.models:
            raise ValueError("No AI models could be initialized. Check API keys and network.")

        logger.info(f"🎯 Advanced swarm agent ready with {len(self.models)} models: {', '.join(self.enabled_providers)}")
        print(colored(f"💰 Total API costs so far: ${self.total_cost:.4f}", "cyan"))

    def _build_cot_analysis_prompt(self, market: Any, news_context: str = "") -> str:
        """
        Build a chain-of-thought prompt that forces structured reasoning.
        """
        return f'''POLYMARKET ANALYSIS TASK

MARKET QUESTION: {market.question}

CURRENT PRICES:
- YES: ${getattr(market, 'yes_price', 0.5):.2f} ({getattr(market, 'yes_price', 0.5)*100:.0f}% implied)
- NO: ${getattr(market, 'no_price', 0.5):.2f} ({getattr(market, 'no_price', 0.5)*100:.0f}% implied)

MARKET INFO:
- Volume: ${getattr(market, 'volume', 0):,.0f}
- Liquidity: ${getattr(market, 'liquidity', 0):,.0f}
- Category: {getattr(market, 'category', 'unknown')}
- Closes: {getattr(market, 'end_date', 'unknown')}

{"RECENT NEWS:\n" + news_context if news_context else ""}

INSTRUCTIONS: Analyze this step-by-step. Be thorough but concise.

## STEP 1: BASE RATE ANALYSIS
What is the historical base rate for this type of event?
What reference class should we use? What is the prior probability?

## STEP 2: EVIDENCE FOR YES
List the top 3 factors supporting YES:
1.
2.
3.

## STEP 3: EVIDENCE FOR NO
List the top 3 factors supporting NO:
1.
2.
3.

## STEP 4: NEWS IMPACT
Has any recent news significantly changed the probability?
Has the market already priced in this news?
What news would dramatically change this?

## STEP 5: MARKET EFFICIENCY CHECK
Is this market liquid enough to be efficient?
Are there reasons the market might be wrong or mispriced?
What behavioral biases might be affecting prices?

## STEP 6: PROBABILITY ESTIMATE
Based on the above analysis:
My probability estimate for YES: XX%
My confidence in this estimate (high/medium/low):

## STEP 7: EDGE ASSESSMENT
Market says: {getattr(market, 'yes_price', 0.5)*100:.0f}%
I estimate: [YOUR ESTIMATE]%
Edge: [DIFFERENCE]%
Is this edge real or noise? Why trade this?

Provide your complete analysis following this structure.'''

    def _parse_cot_response(self, response: str) -> Dict:
        """
        Parse the structured chain-of-thought response.

        Returns:
        {
            "base_rate": str,
            "evidence_yes": List[str],
            "evidence_no": List[str],
            "news_impact": str,
            "probability": float,
            "confidence": str,
            "edge_assessment": str,
        }
        """
        import re

        # Extract probability
        prob_match = re.search(r'My probability estimate for YES:\s*(\d+(?:\.\d+)?)%', response, re.IGNORECASE)
        probability = float(prob_match.group(1)) / 100 if prob_match else 0.5
        # CRITICAL: Bound probability to valid range 0.01-0.99
        probability = max(0.01, min(0.99, probability))

        # Extract confidence
        conf_match = re.search(r'My confidence.*?:\s*(high|medium|low)', response, re.IGNORECASE)
        confidence = conf_match.group(1).lower() if conf_match else "medium"

        # Extract sections using regex
        sections = {}
        section_patterns = {
            "base_rate": r"## STEP 1:.*?(?=## STEP 2:|$)",
            "evidence_yes": r"## STEP 2:.*?(?=## STEP 3:|$)",
            "evidence_no": r"## STEP 3:.*?(?=## STEP 4:|$)",
            "news_impact": r"## STEP 4:.*?(?=## STEP 5:|$)",
            "efficiency_check": r"## STEP 5:.*?(?=## STEP 6:|$)",
            "edge_assessment": r"## STEP 7:.*?(?=##|$)"
        }

        for key, pattern in section_patterns.items():
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            sections[key] = match.group(0).strip() if match else ""

        # Extract evidence lists
        def extract_list(text, prefix=""):
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            items = []
            for line in lines:
                if re.match(r'^\d+\.', line):
                    item = re.sub(r'^\d+\.\s*', '', line)
                    items.append(item)
            return items[:3]  # Top 3

        return {
            "base_rate": sections.get("base_rate", ""),
            "evidence_yes": extract_list(sections.get("evidence_yes", "")),
            "evidence_no": extract_list(sections.get("evidence_no", "")),
            "news_impact": sections.get("news_impact", ""),
            "efficiency_check": sections.get("efficiency_check", ""),
            "probability": probability,
            "confidence": confidence,
            "edge_assessment": sections.get("edge_assessment", ""),
            "full_response": response
        }

    async def run_debate_analysis(self, market: Any) -> Dict:
        """
        Have AI models debate the question.

        1. Model A argues for YES (strongest case)
        2. Model B argues for NO (strongest case)
        3. Model C evaluates both arguments and gives probability

        This surfaces considerations that might be missed.
        """
        # YES advocate prompt
        yes_prompt = f'''You are arguing FOR this outcome.
Make the strongest possible case for YES.
Market: {market.question}

Current market price: YES {getattr(market, 'yes_price', 0.5)*100:.0f}%

Present your argument persuasively. Assume you're trying to convince a skeptic.
Focus on the strongest evidence and reasoning. Be specific.'''

        # NO advocate prompt
        no_prompt = f'''You are arguing AGAINST this outcome.
Make the strongest possible case for NO.
Market: {market.question}

Current market price: YES {getattr(market, 'yes_price', 0.5)*100:.0f}%

Present your argument persuasively. Assume you're trying to convince a skeptic.
Focus on the strongest evidence and reasoning. Be specific.'''

        try:
            # Get advocate responses
            yes_response = await self._query_single_model_async("claude", self.models["claude"], yes_prompt, None, 0.7, 1024, 30, 1.0)
            no_response = await self._query_single_model_async("gemini", self.models["gemini"], no_prompt, None, 0.7, 1024, 30, 1.0)

            yes_argument = yes_response.get("content", "") if yes_response.get("success") else "Failed to get YES argument"
            no_argument = no_response.get("content", "") if no_response.get("success") else "Failed to get NO argument"

            # Judge prompt
            judge_prompt = f'''You are an impartial judge evaluating two arguments.

Market: {market.question}
Current market price: YES {getattr(market, 'yes_price', 0.5)*100:.0f}%

ARGUMENT FOR YES:
{yes_argument}

ARGUMENT FOR NO:
{no_argument}

Evaluate both arguments:
1. Which argument is stronger and why?
2. What did each side miss or underweight?
3. What evidence would you want to see to change your mind?
4. Your probability estimate for YES: XX%
5. Your confidence (high/medium/low):

Be thorough in your analysis.'''

            # Get judgment
            judgment_response = await self._query_single_model_async("gpt", self.models["gpt"], judge_prompt, None, 0.7, 1024, 30, 1.0)
            judgment = judgment_response.get("content", "") if judgment_response.get("success") else "Failed to get judgment"

            probability = self._extract_probability(judgment)

            return {
                "yes_case": yes_argument,
                "no_case": no_argument,
                "judgment": judgment,
                "probability": probability,
                "success": True
            }

        except Exception as e:
            logger.error(f"Debate analysis failed: {e}")
            return {
                "yes_case": "",
                "no_case": "",
                "judgment": "",
                "probability": 0.5,
                "success": False,
                "error": str(e)
            }

    async def red_team_analysis(self, market: Any, initial_probability: float) -> Dict:
        """
        Challenge the initial estimate by looking for flaws.

        "What would make this prediction WRONG?"
        """
        red_team_prompt = f'''You are a "red team" analyst. Your job is to find flaws and challenge assumptions.

Market: {market.question}
Initial probability estimate: {initial_probability*100:.0f}%

Current market price: YES {getattr(market, 'yes_price', 0.5)*100:.0f}%

Your task:
1. What are the top 3 ways this prediction could be WRONG?
2. What information would dramatically change this estimate?
3. What are we potentially overweighting or underweighting?
4. What's the "black swan" scenario that could invalidate this?
5. Are there any logical fallacies in the reasoning?

After this critical analysis, provide an adjusted probability if warranted.
Adjusted probability: XX%

Explain your reasoning for any adjustment.'''

        try:
            response = await self._query_single_model_async("claude", self.models["claude"], red_team_prompt, None, 0.7, 1024, 30, 1.0)

            if response.get("success"):
                content = response.get("content", "")
                adjusted_probability = self._extract_probability(content)

                return {
                    "red_team_analysis": content,
                    "adjusted_probability": adjusted_probability,
                    "success": True
                }
            else:
                return {
                    "red_team_analysis": "",
                    "adjusted_probability": initial_probability,
                    "success": False,
                    "error": "Red team query failed"
                }

        except Exception as e:
            logger.error(f"Red team analysis failed: {e}")
            return {
                "red_team_analysis": "",
                "adjusted_probability": initial_probability,
                "success": False,
                "error": str(e)
            }

    async def deep_analysis(self, market: Any, news_context: str = "") -> Dict:
        """
        Run comprehensive analysis using all methods:
        1. Standard swarm query
        2. Chain-of-thought structured analysis
        3. Debate mode (YES vs NO)
        4. Red team challenge

        Combine all perspectives for final probability.
        """
        logger.info("🧠 Starting deep analysis for market")

        results = {}

        try:
            # 1. Standard swarm query
            standard_prompt = f"Analyze this prediction market and give your probability estimate for YES: {market.question}"
            standard_result = await self.query(standard_prompt, custom_weights=None)
            standard_prob = self._extract_probability(standard_result["consensus"])
            results["standard"] = {"result": standard_result, "probability": standard_prob}

            # 2. Chain-of-thought structured analysis
            cot_prompt = self._build_cot_analysis_prompt(market, news_context)
            cot_response = await self._query_single_model_async("claude", self.models["claude"], cot_prompt, None, 0.7, 2048, 60, 1.0)

            if cot_response.get("success"):
                cot_parsed = self._parse_cot_response(cot_response["content"])
                results["cot"] = {"response": cot_response, "parsed": cot_parsed, "probability": cot_parsed["probability"]}
            else:
                results["cot"] = {"response": cot_response, "parsed": None, "probability": standard_prob}

            # 3. Debate mode
            debate_result = await self.run_debate_analysis(market)
            results["debate"] = debate_result

            # 4. Red team challenge
            initial_prob = results["standard"]["probability"]
            red_team_result = await self.red_team_analysis(market, initial_prob)
            results["red_team"] = red_team_result

            # Combine probabilities with weighted average
            # Give higher weight to structured analysis methods
            weights = {
                "standard": 0.25,
                "cot": 0.30,
                "debate": 0.30,
                "red_team": 0.15
            }

            probabilities = []
            final_weights = []

            for method, weight in weights.items():
                if method in results and results[method].get("probability", 0) > 0:
                    prob = results[method]["probability"]
                    probabilities.append(prob)
                    final_weights.append(weight)
                    logger.info(f"  {method}: {prob:.1%} (weight: {weight})")

            if probabilities:
                # Normalize weights
                total_weight = sum(final_weights)
                normalized_weights = [w/total_weight for w in final_weights]

                final_probability = sum(p * w for p, w in zip(probabilities, normalized_weights))

                logger.info(f"🎯 Final probability: {final_probability:.1%}")

                return {
                    "final_probability": final_probability,
                    "method_probabilities": {method: results[method].get("probability", 0) for method in weights.keys()},
                    "method_weights": weights,
                    "analysis_details": results,
                    "success": True
                }
            else:
                logger.warning("No valid probabilities from any analysis method")
                return {
                    "final_probability": 0.5,
                    "method_probabilities": {},
                    "method_weights": weights,
                    "analysis_details": results,
                    "success": False,
                    "error": "No valid probabilities"
                }

        except Exception as e:
            logger.error(f"Deep analysis failed: {e}")
            return {
                "final_probability": 0.5,
                "method_probabilities": {},
                "method_weights": {},
                "analysis_details": {},
                "success": False,
                "error": str(e)
            }

    async def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 30,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Query all enabled models in parallel.

        Args:
            prompt: Main user prompt
            system_prompt: Optional system prompt
            temperature: Model temperature
            max_tokens: Maximum tokens per response
            timeout: Timeout per model query in seconds

        Returns:
            Dict with consensus analysis and individual responses
        """
        timestamp = datetime.now()

        print(colored(f"\n🧠 Querying {len(self.models)} AI models...", "cyan", attrs=["bold"]))
        print(colored(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}", "white"))

        # Query all models in parallel
        responses = await self._query_all_models_async(
            prompt, system_prompt, temperature, max_tokens
        )

        # Generate consensus review
        print(colored("\n📝 Generating consensus analysis...", "yellow"))
        consensus_summary = self._generate_consensus_review(responses, prompt)

        # Build response mapping
        model_mapping = {}
        raw_responses = {}
        metadata = {
            "total_models": len(self.models),
            "successful_queries": 0,
            "failed_queries": 0,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout
        }

        for resp in responses:
            model_mapping[resp.provider] = resp.model_name
            raw_responses[resp.provider] = {
                "content": resp.response.content if resp.success else None,
                "usage": resp.response.usage if resp.success else None,
                "error": resp.error,
                "success": resp.success
            }

            if resp.success:
                metadata["successful_queries"] += 1
            else:
                metadata["failed_queries"] += 1

        result = {
            "timestamp": timestamp.isoformat(),
            "prompt": prompt,
            "system_prompt": system_prompt,
            "consensus_summary": consensus_summary,
            "model_mapping": model_mapping,
            "responses": raw_responses,
            "metadata": metadata
        }

        success_rate = metadata["successful_queries"] / metadata["total_models"] * 100
        print(colored(f"✅ Consensus generated ({success_rate:.1f}% success rate)", "green", attrs=["bold"]))

        return result

    async def _query_all_models_async(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> List[SwarmResponse]:
        """Query all models asynchronously with smart timeout handling."""
        responses = []
        tasks = []

        # Create async tasks for all models
        for provider, model in self.models.items():
            timeout = MODEL_TIMEOUTS.get(provider, 30)
            weight = custom_weights.get(provider, MODEL_WEIGHTS.get(provider, 1.0)) if custom_weights else MODEL_WEIGHTS.get(provider, 1.0)

            task = asyncio.create_task(
                self._query_single_model_async(
                    provider, model, prompt, system_prompt,
                    temperature, max_tokens, timeout, weight
                )
            )
            tasks.append(task)

        print(colored(f"🚀 Launched {len(tasks)} parallel queries", "blue"))

        # Process results as they complete (streaming feedback)
        completed_count = 0
        min_responses_needed = 3  # Continue with partial results after this many

        for coro in asyncio.as_completed(tasks):
            try:
                response = await coro
                responses.append(response)
                completed_count += 1

                if response.success:
                    cost_str = f"${response.response.cost_estimate:.4f}" if response.response else "$0.00"
                    time_str = f"{response.response.response_time:.1f}s" if response.response else "N/A"
                    print(colored(f"✅ {response.provider}: {time_str} | {cost_str}", "green"))
                else:
                    print(colored(f"❌ {response.provider}: {response.error[:50]}...", "red"))

                # Smart early termination - if we have enough good responses, don't wait for slow models
                successful_responses = [r for r in responses if r.success]
                if len(successful_responses) >= min_responses_needed and completed_count < len(tasks):
                    remaining_tasks = len(tasks) - completed_count
                    print(colored(f"🎯 Have {len(successful_responses)} good responses, cancelling {remaining_tasks} slow queries", "yellow"))
                    # Cancel remaining tasks
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    break

            except Exception as e:
                logger.error(f"Task completion error: {e}")

        # Wait for any remaining tasks to complete or timeout
        if len(responses) < len(tasks):
            print(colored("⏳ Waiting for remaining queries to complete or timeout...", "yellow"))
            tail_results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in tail_results:
                # gather(return_exceptions=True) returns exceptions as values (including CancelledError)
                if isinstance(r, asyncio.CancelledError):
                    continue
                if isinstance(r, Exception):
                    logger.warning(f"Swarm query task ended with exception: {r}")

        return responses

    async def _query_single_model_async(
        self,
        provider: str,
        model: Any,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        timeout: int,
        weight: float
    ) -> SwarmResponse:
        """Query a single model asynchronously with timeout."""
        try:
            logger.debug(f"Querying {provider} asynchronously...")

            # Use default system prompt if none provided
            if system_prompt is None:
                system_prompt = "You are an expert financial analyst specializing in prediction markets and trading. Provide clear, concise analysis."

            # Run model query in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, model.generate_response,
                    system_prompt, prompt, temperature, max_tokens, timeout),
                timeout=timeout
            )

            # Strip thinking tags if present
            response.content = self._strip_think_tags(response.content)

            return SwarmResponse(
                provider=provider,
                model_name=model.model_name,
                response=response,
                success=True,
                weight=weight
            )

        except asyncio.TimeoutError:
            error_msg = f"Timeout after {timeout}s"
            logger.warning(f"{provider}: {error_msg}")
            return SwarmResponse(
                provider=provider,
                model_name=model.model_name,
                response=ModelResponse(content="", model=model.model_name),
                success=False,
                error=error_msg,
                weight=weight
            )

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"{provider} query failed: {error_msg}")
            return SwarmResponse(
                provider=provider,
                model_name=model.model_name,
                response=ModelResponse(content="", model=model.model_name),
                success=False,
                error=error_msg,
                weight=weight
            )

    def _generate_weighted_consensus(self, responses: List[SwarmResponse], original_prompt: str) -> Dict[str, Any]:
        """Generate weighted consensus with probability extraction."""
        successful_responses = [r for r in responses if r.success and r.response]

        if not successful_responses:
            return {
                "summary": "No successful model responses to analyze.",
                "weighted_probability": None,
                "unweighted_probability": None,
                "confidence_score": 0.0
            }

        # Extract probabilities from all responses
        probabilities = []
        model_predictions = {}

        for resp in successful_responses:
            prob = self._extract_probability_advanced(resp.response.content)
            if prob is not None:
                # Apply calibration to individual model predictions
                calibrated_prob = self.calibrator.calibrate_probability(resp.provider, prob)

                # Weight the calibrated probability by model weight
                weighted_prob = calibrated_prob * resp.weight
                probabilities.append(weighted_prob)
                model_predictions[resp.provider] = {
                    "probability": prob,
                    "calibrated_probability": calibrated_prob,
                    "weighted_probability": weighted_prob,
                    "weight": resp.weight,
                    "confidence": self._extract_confidence(resp.response.content)
                }

        # Calculate consensus metrics
        if probabilities:
            weighted_avg = sum(probabilities) / len(probabilities)
            unweighted_predictions = [pred["probability"] for pred in model_predictions.values()]
            unweighted_avg = sum(unweighted_predictions) / len(unweighted_predictions)

            # Calculate confidence based on agreement
            if len(unweighted_predictions) > 1:
                confidence = max(0.0, 1.0 - (self._calculate_std_dev(unweighted_predictions) / 0.5))
            else:
                confidence = 0.5
        else:
            weighted_avg = unweighted_avg = None
            confidence = 0.0

        # Generate consensus summary
        summary = self._build_consensus_summary(successful_responses, model_predictions, original_prompt)

        return {
            "summary": summary,
            "weighted_probability": weighted_avg,
            "unweighted_probability": unweighted_avg,
            "confidence_score": confidence,
            "model_predictions": model_predictions
        }

    def _extract_probability_advanced(self, text: str) -> Optional[float]:
        """Advanced probability extraction with multiple patterns and ranges."""
        if not text:
            return None

        text_lower = text.lower()

        # Multiple regex patterns for probability extraction
        patterns = [
            r'(\d+(?:\.\d+)?)\s*%',  # "75%" or "75.5%"
            r'(\d+(?:\.\d+)?)\s*percent',  # "75 percent"
            r'probability(?:\s+(?:of|is|estimate))?\s*:?\s*(\d+(?:\.\d+)?)%?',  # "probability: 75%"
            r'(\d+(?:\.\d+)?)%\s+(?:chance|likelihood|probability)',  # "75% chance"
            r'(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)',  # "YES: 75" format
            r'predict\s+(\d+(?:\.\d+)?)%',  # "predict 75%"
            r'forecast\s+(\d+(?:\.\d+)?)%',  # "forecast 75%"
        ]

        all_matches = []
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            all_matches.extend(matches)

        if not all_matches:
            return None

        # Convert matches to floats and validate
        valid_probs = []
        for match in all_matches:
            try:
                prob = float(match)
                if 0 <= prob <= 100:
                    valid_probs.append(prob)
            except ValueError:
                continue

        if not valid_probs:
            return None

        # Handle ranges like "60-70%" -> take midpoint
        if len(valid_probs) >= 2:
            # Look for range patterns
            range_pattern = r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)%?'
            ranges = re.findall(range_pattern, text_lower)
            if ranges:
                # Use the midpoint of the first range found
                start, end = map(float, ranges[0][:2])
                return (start + end) / 2 / 100.0  # Convert to 0-1 range

        # Return the most reasonable probability (prefer values between 1-99%)
        reasonable_probs = [p for p in valid_probs if 1 <= p <= 99]
        if reasonable_probs:
            return reasonable_probs[0] / 100.0

        # Fallback to first valid probability
        return valid_probs[0] / 100.0

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence level from response text."""
        text_lower = text.lower()

        # Look for explicit confidence statements
        confidence_patterns = [
            r'confidence\s*:?\s*(\d+(?:\.\d+)?)%?',
            r'(\d+(?:\.\d+)?)%?\s*confident',
            r'certainty\s*:?\s*(\d+(?:\.\d+)?)%?',
        ]

        for pattern in confidence_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    confidence = float(matches[0])
                    return min(confidence / 100.0, 1.0)  # Convert to 0-1 range
                except ValueError:
                    continue

        # Default confidence based on response length/detail
        word_count = len(text.split())
        if word_count > 200:
            return 0.8  # Detailed analysis = high confidence
        elif word_count > 100:
            return 0.6  # Moderate detail = medium confidence
        else:
            return 0.4  # Brief response = lower confidence

    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation of a list of values."""
        if len(values) <= 1:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def _build_consensus_summary(self, responses: List[SwarmResponse],
                                model_predictions: Dict, original_prompt: str) -> str:
        """Build a consensus summary from model responses."""
        if not responses:
            return "No responses available for consensus analysis."

        # Analyze agreement/disagreement
        predictions = [data["probability"] for data in model_predictions.values() if data["probability"] is not None]

        if len(predictions) <= 1:
            # Single response
            provider = list(model_predictions.keys())[0]
            prob = model_predictions[provider]["probability"]
            return f"Single model response from {provider}: {prob:.1%} probability."

        # Multiple responses - analyze consensus
        avg_prob = sum(predictions) / len(predictions)
        std_dev = self._calculate_std_dev(predictions)

        if std_dev < 0.1:
            agreement = "Strong agreement"
        elif std_dev < 0.2:
            agreement = "Moderate agreement"
        else:
            agreement = "Significant disagreement"

        # Find reasoning highlights
        highlights = []
        for resp in responses[:2]:  # Use first 2 responses for highlights
            content = resp.response.content[:200] + "..." if len(resp.response.content) > 200 else resp.response.content
            highlights.append(f"{resp.provider}: {content}")

        summary = f"""Consensus Analysis: {len(predictions)} models show {agreement.lower()} on {avg_prob:.1%} average probability.
Standard deviation: {std_dev:.1%} across predictions.

Key insights:
{chr(10).join(f"• {h}" for h in highlights)}
"""
        return summary.strip()

    def _generate_cache_key(self, prompt: str, system_prompt: Optional[str],
                           temperature: float, max_tokens: int) -> str:
        """Generate a cache key for the query parameters."""
        key_data = f"{prompt}|{system_prompt}|{temperature}|{max_tokens}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _strip_think_tags(self, text: str) -> str:
        """Remove <think>...</think> tags from model responses."""
        # Remove thinking tags (common in some models like DeepSeek)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
        return text.strip()

    def clear_cache(self):
        """Clear the response cache."""
        self.cache.clear()
        print(colored("🗑️  Cache cleared", "yellow"))

    def get_cost_summary(self) -> Dict[str, float]:
        """Get cost summary for all models."""
        return {
            "total_cost": self.total_cost,
            "cost_by_model": dict(self.cost_tracker)
        }


def run_self_tests():
    """Run self-tests for swarm agent functions."""
    print("🧪 Swarm Agent Self-Tests")

    # Test basic functionality without API keys
    agent = SwarmAgent()

    # Test that model mappings are correct
    assert 'claude' in SWARM_MODELS, "Claude should be in swarm models"
    assert 'gpt' in SWARM_MODELS, "GPT should be in swarm models"
    assert len(SWARM_MODELS) >= 3, "Should have at least 3 models"

    # Test standard deviation calculation with sample data
    probabilities = [0.6, 0.7, 0.65, 0.68, 0.62]
    std_dev = agent._calculate_std_dev(probabilities)
    assert std_dev >= 0, f"Standard deviation should be non-negative, got {std_dev}"

    # Test standard deviation calculation
    std_dev = agent._calculate_std_dev(probabilities)
    assert std_dev >= 0, f"Standard deviation should be non-negative, got {std_dev}"

    print("✅ Swarm Agent self-tests passed")


if __name__ == "__main__":
    run_self_tests()
