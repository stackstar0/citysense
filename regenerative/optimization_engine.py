"""
Optimization Engine - Self-healing and adaptive urban optimization
Implements regenerative algorithms for urban system optimization
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import statistics
import math
from dataclasses import dataclass, asdict
from enum import Enum
import json

logger = logging.getLogger(__name__)

class OptimizationType(Enum):
    ENERGY_EFFICIENCY = "energy_efficiency"
    TRAFFIC_FLOW = "traffic_flow"
    AIR_QUALITY = "air_quality"
    RESOURCE_ALLOCATION = "resource_allocation"
    ECOSYSTEM_HEALTH = "ecosystem_health"
    SOCIAL_EQUITY = "social_equity"
    RESILIENCE_BUILDING = "resilience_building"

class OptimizationPriority(Enum):
    EMERGENCY = "emergency"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MAINTENANCE = "maintenance"

@dataclass
class OptimizationAction:
    """Represents a specific optimization action"""
    action_id: str
    optimization_type: OptimizationType
    priority: OptimizationPriority
    description: str

    # Target systems and parameters
    target_systems: List[str]
    parameters: Dict[str, Any]

    # Expected outcomes
    expected_impact: Dict[str, float]  # Domain -> impact score
    confidence: float  # 0-1

    # Implementation details
    implementation_time_minutes: int
    resource_requirements: Dict[str, Any]
    prerequisites: List[str]

    # Monitoring
    success_metrics: List[str]
    rollback_plan: Optional[str]

    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class OptimizationResult:
    """Result of an optimization action"""
    action_id: str
    status: str  # "success", "partial", "failed"
    actual_impact: Dict[str, float]
    implementation_duration_minutes: int
    success_score: float  # 0-1
    lessons_learned: List[str]
    timestamp: datetime

class OptimizationEngine:
    """
    Self-healing optimization engine for urban systems
    Implements regenerative algorithms that learn and adapt
    """

    def __init__(self):
        self.active_optimizations: List[OptimizationAction] = []
        self.optimization_history: List[OptimizationResult] = []

        # Learning parameters
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.7
        self.success_threshold = 0.8

        # Optimization strategies
        self.strategies = self._initialize_optimization_strategies()

        # System state memory for learning
        self.system_state_memory: List[Dict[str, Any]] = []
        self.max_memory_size = 1000

        logger.info("🔄 Optimization engine initialized with regenerative algorithms")

    def _initialize_optimization_strategies(self) -> Dict[OptimizationType, Dict[str, Any]]:
        """Initialize optimization strategies for different system types"""

        return {
            OptimizationType.ENERGY_EFFICIENCY: {
                "base_algorithm": "load_balancing_with_renewables",
                "parameters": {
                    "renewable_priority_weight": 1.5,
                    "load_balancing_threshold": 0.8,
                    "peak_shaving_aggressive": True
                },
                "success_metrics": ["energy_consumption_reduction", "renewable_ratio_increase", "grid_stability"],
                "typical_impact_range": (5, 25),  # % improvement
                "implementation_complexity": "medium"
            },

            OptimizationType.TRAFFIC_FLOW: {
                "base_algorithm": "adaptive_signal_timing",
                "parameters": {
                    "flow_optimization_weight": 1.0,
                    "emission_reduction_weight": 0.8,
                    "pedestrian_priority_zones": True
                },
                "success_metrics": ["average_speed_increase", "congestion_reduction", "emission_decrease"],
                "typical_impact_range": (10, 40),
                "implementation_complexity": "high"
            },

            OptimizationType.AIR_QUALITY: {
                "base_algorithm": "multi_source_pollution_control",
                "parameters": {
                    "traffic_reduction_priority": 1.2,
                    "industrial_regulation_weight": 1.0,
                    "green_space_enhancement": 0.9
                },
                "success_metrics": ["pm25_reduction", "no2_reduction", "overall_aqi_improvement"],
                "typical_impact_range": (8, 30),
                "implementation_complexity": "high"
            },

            OptimizationType.RESOURCE_ALLOCATION: {
                "base_algorithm": "dynamic_resource_optimization",
                "parameters": {
                    "demand_prediction_horizon": 24,  # hours
                    "efficiency_weight": 1.1,
                    "equity_consideration": 0.9
                },
                "success_metrics": ["resource_utilization_efficiency", "service_level_improvement", "cost_reduction"],
                "typical_impact_range": (12, 35),
                "implementation_complexity": "medium"
            },

            OptimizationType.ECOSYSTEM_HEALTH: {
                "base_algorithm": "ecological_restoration_optimization",
                "parameters": {
                    "biodiversity_weight": 1.3,
                    "water_quality_priority": 1.1,
                    "carbon_sequestration_bonus": 1.2
                },
                "success_metrics": ["biodiversity_index", "water_quality_improvement", "carbon_absorption"],
                "typical_impact_range": (5, 20),
                "implementation_complexity": "low"
            },

            OptimizationType.SOCIAL_EQUITY: {
                "base_algorithm": "equitable_service_distribution",
                "parameters": {
                    "access_fairness_weight": 1.4,
                    "quality_consistency_weight": 1.0,
                    "community_input_factor": 1.1
                },
                "success_metrics": ["service_access_equality", "satisfaction_scores", "community_engagement"],
                "typical_impact_range": (8, 25),
                "implementation_complexity": "high"
            },

            OptimizationType.RESILIENCE_BUILDING: {
                "base_algorithm": "adaptive_resilience_enhancement",
                "parameters": {
                    "redundancy_building_weight": 1.2,
                    "adaptability_enhancement": 1.1,
                    "recovery_speed_optimization": 1.3
                },
                "success_metrics": ["system_redundancy", "adaptation_capacity", "recovery_time"],
                "typical_impact_range": (10, 40),
                "implementation_complexity": "high"
            }
        }

    async def optimize_city_systems(self, vital_signs, insights: List) -> List[OptimizationAction]:
        """
        Generate optimization actions based on city vital signs and insights
        """

        logger.info("🔄 Generating optimization actions for city systems")

        optimization_actions = []

        try:
            # Store current state for learning
            current_state = {
                "vital_signs": vital_signs.to_dict() if vital_signs else {},
                "insights": [asdict(insight) for insight in insights],
                "timestamp": datetime.now()
            }
            self._store_system_state(current_state)

            # Analyze critical issues first
            critical_actions = await self._generate_critical_optimizations(vital_signs, insights)
            optimization_actions.extend(critical_actions)

            # Generate proactive optimizations
            proactive_actions = await self._generate_proactive_optimizations(vital_signs)
            optimization_actions.extend(proactive_actions)

            # Generate learning-based optimizations
            learning_actions = await self._generate_learning_optimizations()
            optimization_actions.extend(learning_actions)

            # Prioritize and filter actions
            prioritized_actions = self._prioritize_optimizations(optimization_actions)

            # Add to active optimizations
            self.active_optimizations.extend(prioritized_actions)

            logger.info(f"Generated {len(prioritized_actions)} optimization actions")

            return prioritized_actions

        except Exception as e:
            logger.error(f"Error generating optimizations: {e}")
            return []

    async def _generate_critical_optimizations(self, vital_signs, insights: List) -> List[OptimizationAction]:
        """Generate optimizations for critical issues"""

        actions = []

        if not vital_signs:
            return actions

        # Critical air quality optimization
        if vital_signs.air_quality_index < 30:
            action = OptimizationAction(
                action_id=f"critical_air_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                optimization_type=OptimizationType.AIR_QUALITY,
                priority=OptimizationPriority.EMERGENCY,
                description="Emergency air quality improvement - activate all pollution reduction systems",
                target_systems=["traffic_control", "industrial_emissions", "air_purification"],
                parameters={
                    "traffic_reduction_zones": ["downtown", "industrial_district"],
                    "traffic_reduction_percentage": 40,
                    "industrial_emission_limits": "emergency_level",
                    "air_purification_intensity": "maximum",
                    "public_transport_boost": 50
                },
                expected_impact={
                    "air_quality_improvement": 25,
                    "public_health_benefit": 20,
                    "economic_cost": -15
                },
                confidence=0.85,
                implementation_time_minutes=30,
                resource_requirements={
                    "emergency_response_teams": 3,
                    "budget_allocation": 50000,
                    "system_capacity": "high"
                },
                prerequisites=["emergency_protocols_activated", "stakeholder_notification"],
                success_metrics=["pm25_reduction_rate", "no2_levels", "public_health_indicators"],
                rollback_plan="gradual_return_to_normal_operations",
                timestamp=datetime.now(),
                metadata={"trigger": "critical_air_quality", "urgency": "immediate"}
            )
            actions.append(action)

        # Critical resilience optimization
        if vital_signs.resilience_score < 40:
            action = OptimizationAction(
                action_id=f"critical_resilience_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                optimization_type=OptimizationType.RESILIENCE_BUILDING,
                priority=OptimizationPriority.EMERGENCY,
                description="Emergency resilience enhancement - activate backup systems and redundancies",
                target_systems=["energy_grid", "water_systems", "communication_networks", "emergency_services"],
                parameters={
                    "backup_systems_activation": "all_critical",
                    "redundancy_level": "maximum",
                    "emergency_reserves_deployment": 80,
                    "cross_system_coordination": "enhanced"
                },
                expected_impact={
                    "resilience_improvement": 35,
                    "system_stability": 25,
                    "emergency_preparedness": 40
                },
                confidence=0.80,
                implementation_time_minutes=60,
                resource_requirements={
                    "emergency_coordination_center": 1,
                    "technical_teams": 5,
                    "emergency_budget": 100000
                },
                prerequisites=["crisis_response_activated", "inter_agency_coordination"],
                success_metrics=["system_redundancy_active", "response_time_improvement", "stability_indicators"],
                rollback_plan="phased_return_to_standard_operations",
                timestamp=datetime.now(),
                metadata={"trigger": "critical_resilience", "scope": "citywide"}
            )
            actions.append(action)

        # Critical insights-based optimizations
        critical_insights = [insight for insight in insights if insight.priority == "critical"]

        for insight in critical_insights:
            if insight.category == "climate":
                action = self._create_climate_optimization_action(insight, OptimizationPriority.HIGH)
                if action:
                    actions.append(action)
            elif insight.category == "infrastructure":
                action = self._create_infrastructure_optimization_action(insight, OptimizationPriority.HIGH)
                if action:
                    actions.append(action)

        return actions

    async def _generate_proactive_optimizations(self, vital_signs) -> List[OptimizationAction]:
        """Generate proactive optimizations to improve system performance"""

        actions = []

        if not vital_signs:
            return actions

        # Energy efficiency optimization
        if vital_signs.energy_efficiency < 70:
            strategy = self.strategies[OptimizationType.ENERGY_EFFICIENCY]

            # Calculate optimization parameters based on current state
            renewable_boost = max(10, (70 - vital_signs.energy_efficiency) * 0.5)

            action = OptimizationAction(
                action_id=f"energy_efficiency_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                optimization_type=OptimizationType.ENERGY_EFFICIENCY,
                priority=OptimizationPriority.MEDIUM,
                description=f"Improve energy efficiency by {renewable_boost:.1f}% through smart grid optimization",
                target_systems=["smart_grid", "renewable_sources", "energy_storage", "demand_management"],
                parameters={
                    "renewable_priority_boost": renewable_boost,
                    "load_balancing_optimization": True,
                    "demand_response_activation": "moderate",
                    "storage_optimization": "active"
                },
                expected_impact={
                    "energy_efficiency_improvement": renewable_boost,
                    "cost_savings": renewable_boost * 0.8,
                    "carbon_emission_reduction": renewable_boost * 1.2
                },
                confidence=0.75,
                implementation_time_minutes=120,
                resource_requirements={"grid_operators": 2, "system_analysts": 1},
                prerequisites=["grid_stability_check", "renewable_capacity_assessment"],
                success_metrics=strategy["success_metrics"],
                rollback_plan="return_to_previous_grid_configuration",
                timestamp=datetime.now(),
                metadata={"optimization_target": renewable_boost, "baseline_efficiency": vital_signs.energy_efficiency}
            )
            actions.append(action)

        # Traffic flow optimization
        if vital_signs.traffic_flow_rate < 60:
            flow_improvement_target = min(30, (60 - vital_signs.traffic_flow_rate) * 0.8)

            action = OptimizationAction(
                action_id=f"traffic_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                optimization_type=OptimizationType.TRAFFIC_FLOW,
                priority=OptimizationPriority.MEDIUM,
                description=f"Optimize traffic flow to improve efficiency by {flow_improvement_target:.1f}%",
                target_systems=["traffic_signals", "route_optimization", "public_transport", "parking_management"],
                parameters={
                    "signal_timing_optimization": "adaptive",
                    "route_guidance_enhancement": True,
                    "public_transport_priority": 1.2,
                    "dynamic_pricing_parking": "moderate"
                },
                expected_impact={
                    "traffic_flow_improvement": flow_improvement_target,
                    "travel_time_reduction": flow_improvement_target * 0.6,
                    "emission_reduction": flow_improvement_target * 0.4
                },
                confidence=0.70,
                implementation_time_minutes=90,
                resource_requirements={"traffic_engineers": 2, "system_updates": "moderate"},
                prerequisites=["traffic_pattern_analysis", "signal_system_check"],
                success_metrics=["average_speed_improvement", "congestion_index_reduction", "public_satisfaction"],
                rollback_plan="restore_previous_signal_timing",
                timestamp=datetime.now(),
                metadata={"target_improvement": flow_improvement_target, "baseline_flow": vital_signs.traffic_flow_rate}
            )
            actions.append(action)

        # Ecological health optimization
        if vital_signs.ecological_health < 65:
            ecological_improvement = min(25, (65 - vital_signs.ecological_health) * 0.6)

            action = OptimizationAction(
                action_id=f"ecological_health_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                optimization_type=OptimizationType.ECOSYSTEM_HEALTH,
                priority=OptimizationPriority.LOW,
                description=f"Enhance ecological health through targeted interventions",
                target_systems=["green_spaces", "water_systems", "waste_management", "biodiversity_programs"],
                parameters={
                    "green_space_enhancement": ecological_improvement,
                    "water_quality_programs": "accelerated",
                    "waste_reduction_initiatives": 1.3,
                    "biodiversity_protection": "enhanced"
                },
                expected_impact={
                    "ecological_health_improvement": ecological_improvement,
                    "biodiversity_increase": ecological_improvement * 0.7,
                    "community_wellbeing": ecological_improvement * 0.5
                },
                confidence=0.65,
                implementation_time_minutes=240,  # Longer-term intervention
                resource_requirements={"environmental_teams": 3, "community_engagement": "high"},
                prerequisites=["environmental_assessment", "community_consultation"],
                success_metrics=["biodiversity_index", "water_quality_metrics", "green_space_utilization"],
                rollback_plan="gradual_scale_back_if_ineffective",
                timestamp=datetime.now(),
                metadata={"target_improvement": ecological_improvement, "baseline_health": vital_signs.ecological_health}
            )
            actions.append(action)

        return actions

    async def _generate_learning_optimizations(self) -> List[OptimizationAction]:
        """Generate optimizations based on historical learning"""

        actions = []

        if len(self.optimization_history) < 10:  # Need sufficient history
            return actions

        # Analyze successful optimizations
        successful_optimizations = [
            result for result in self.optimization_history
            if result.success_score > self.success_threshold
        ]

        if not successful_optimizations:
            return actions

        # Find patterns in successful optimizations
        success_patterns = self._analyze_success_patterns(successful_optimizations)

        # Generate new optimizations based on successful patterns
        for pattern in success_patterns[:3]:  # Top 3 patterns
            action = self._create_pattern_based_optimization(pattern)
            if action:
                actions.append(action)

        return actions

    def _analyze_success_patterns(self, successful_optimizations: List[OptimizationResult]) -> List[Dict[str, Any]]:
        """Analyze patterns in successful optimizations"""

        patterns = []

        # Group by optimization type
        type_groups = {}
        for result in successful_optimizations:
            # Find the original action
            action = next((a for a in self.active_optimizations if a.action_id == result.action_id), None)
            if action:
                opt_type = action.optimization_type
                if opt_type not in type_groups:
                    type_groups[opt_type] = []
                type_groups[opt_type].append((action, result))

        # Analyze each type
        for opt_type, action_results in type_groups.items():
            if len(action_results) >= 3:  # Need multiple examples
                avg_success = statistics.mean([result.success_score for _, result in action_results])
                avg_impact = {}

                # Calculate average impact
                for domain in ["energy", "traffic", "air_quality", "ecological", "social"]:
                    impacts = []
                    for action, result in action_results:
                        if domain in result.actual_impact:
                            impacts.append(result.actual_impact[domain])
                    if impacts:
                        avg_impact[domain] = statistics.mean(impacts)

                pattern = {
                    "optimization_type": opt_type,
                    "success_rate": avg_success,
                    "average_impact": avg_impact,
                    "sample_size": len(action_results),
                    "recommended_parameters": self._extract_successful_parameters(action_results)
                }
                patterns.append(pattern)

        # Sort by success rate
        patterns.sort(key=lambda p: p["success_rate"], reverse=True)

        return patterns

    def _extract_successful_parameters(self, action_results: List[Tuple]) -> Dict[str, Any]:
        """Extract common parameters from successful optimizations"""

        param_values = {}

        for action, result in action_results:
            for param, value in action.parameters.items():
                if param not in param_values:
                    param_values[param] = []
                param_values[param].append(value)

        # Calculate average/mode for each parameter
        recommended_params = {}
        for param, values in param_values.items():
            if all(isinstance(v, (int, float)) for v in values):
                recommended_params[param] = statistics.mean(values)
            else:
                # For non-numeric values, find most common
                from collections import Counter
                counter = Counter(values)
                recommended_params[param] = counter.most_common(1)[0][0]

        return recommended_params

    def _create_pattern_based_optimization(self, pattern: Dict[str, Any]) -> Optional[OptimizationAction]:
        """Create optimization action based on successful pattern"""

        try:
            opt_type = pattern["optimization_type"]

            action = OptimizationAction(
                action_id=f"pattern_based_{opt_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                optimization_type=opt_type,
                priority=OptimizationPriority.MEDIUM,
                description=f"Pattern-based {opt_type.value} optimization (success rate: {pattern['success_rate']:.1%})",
                target_systems=self._get_target_systems_for_type(opt_type),
                parameters=pattern["recommended_parameters"],
                expected_impact=pattern["average_impact"],
                confidence=min(0.9, pattern["success_rate"]),
                implementation_time_minutes=self._estimate_implementation_time(opt_type),
                resource_requirements=self._get_standard_resource_requirements(opt_type),
                prerequisites=self._get_standard_prerequisites(opt_type),
                success_metrics=self.strategies[opt_type]["success_metrics"],
                rollback_plan=f"revert_to_baseline_{opt_type.value}",
                timestamp=datetime.now(),
                metadata={
                    "pattern_based": True,
                    "historical_success_rate": pattern["success_rate"],
                    "sample_size": pattern["sample_size"]
                }
            )

            return action

        except Exception as e:
            logger.error(f"Error creating pattern-based optimization: {e}")
            return None

    def _prioritize_optimizations(self, actions: List[OptimizationAction]) -> List[OptimizationAction]:
        """Prioritize optimization actions based on impact and feasibility"""

        # Calculate priority scores
        scored_actions = []

        for action in actions:
            # Base priority score
            priority_scores = {
                OptimizationPriority.EMERGENCY: 100,
                OptimizationPriority.HIGH: 80,
                OptimizationPriority.MEDIUM: 60,
                OptimizationPriority.LOW: 40,
                OptimizationPriority.MAINTENANCE: 20
            }

            score = priority_scores[action.priority]

            # Add impact score
            total_expected_impact = sum(abs(impact) for impact in action.expected_impact.values())
            score += total_expected_impact * 0.5

            # Add confidence bonus
            score += action.confidence * 20

            # Subtract complexity penalty
            complexity_penalty = action.implementation_time_minutes * 0.1
            score -= complexity_penalty

            scored_actions.append((score, action))

        # Sort by score (highest first)
        scored_actions.sort(key=lambda x: x[0], reverse=True)

        # Return top 10 actions
        return [action for score, action in scored_actions[:10]]

    def _store_system_state(self, state: Dict[str, Any]):
        """Store system state for learning"""
        self.system_state_memory.append(state)

        # Limit memory size
        if len(self.system_state_memory) > self.max_memory_size:
            self.system_state_memory = self.system_state_memory[-self.max_memory_size:]

    def _create_climate_optimization_action(self, insight, priority: OptimizationPriority) -> Optional[OptimizationAction]:
        """Create climate-related optimization action from insight"""

        try:
            action = OptimizationAction(
                action_id=f"climate_insight_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                optimization_type=OptimizationType.AIR_QUALITY,
                priority=priority,
                description=f"Climate optimization: {insight.title}",
                target_systems=["air_quality_systems", "emission_control", "green_infrastructure"],
                parameters={
                    "intervention_intensity": "high" if priority == OptimizationPriority.HIGH else "moderate",
                    "target_areas": insight.affected_areas
                },
                expected_impact=insight.predicted_impact,
                confidence=insight.confidence_score,
                implementation_time_minutes=60,
                resource_requirements={"environmental_teams": 2},
                prerequisites=["environmental_impact_assessment"],
                success_metrics=["air_quality_improvement", "emission_reduction"],
                rollback_plan="gradual_intervention_reduction",
                timestamp=datetime.now(),
                metadata={"insight_triggered": True, "insight_id": insight.insight_id}
            )

            return action

        except Exception as e:
            logger.error(f"Error creating climate optimization: {e}")
            return None

    def _create_infrastructure_optimization_action(self, insight, priority: OptimizationPriority) -> Optional[OptimizationAction]:
        """Create infrastructure-related optimization action from insight"""

        try:
            action = OptimizationAction(
                action_id=f"infrastructure_insight_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                optimization_type=OptimizationType.RESILIENCE_BUILDING,
                priority=priority,
                description=f"Infrastructure optimization: {insight.title}",
                target_systems=["critical_infrastructure", "backup_systems", "monitoring_systems"],
                parameters={
                    "resilience_enhancement": "comprehensive",
                    "affected_systems": insight.affected_areas
                },
                expected_impact=insight.predicted_impact,
                confidence=insight.confidence_score,
                implementation_time_minutes=180,
                resource_requirements={"infrastructure_teams": 3, "emergency_budget": 75000},
                prerequisites=["infrastructure_assessment", "stakeholder_coordination"],
                success_metrics=["system_reliability", "redundancy_improvement", "response_time"],
                rollback_plan="phased_system_restoration",
                timestamp=datetime.now(),
                metadata={"insight_triggered": True, "insight_id": insight.insight_id}
            )

            return action

        except Exception as e:
            logger.error(f"Error creating infrastructure optimization: {e}")
            return None

    def _get_target_systems_for_type(self, opt_type: OptimizationType) -> List[str]:
        """Get standard target systems for optimization type"""

        system_map = {
            OptimizationType.ENERGY_EFFICIENCY: ["smart_grid", "renewable_sources", "energy_storage"],
            OptimizationType.TRAFFIC_FLOW: ["traffic_signals", "route_optimization", "public_transport"],
            OptimizationType.AIR_QUALITY: ["emission_control", "air_purification", "green_infrastructure"],
            OptimizationType.RESOURCE_ALLOCATION: ["resource_management", "distribution_systems"],
            OptimizationType.ECOSYSTEM_HEALTH: ["green_spaces", "water_systems", "biodiversity_programs"],
            OptimizationType.SOCIAL_EQUITY: ["service_distribution", "community_programs", "accessibility"],
            OptimizationType.RESILIENCE_BUILDING: ["backup_systems", "redundancy", "emergency_response"]
        }

        return system_map.get(opt_type, ["general_systems"])

    def _estimate_implementation_time(self, opt_type: OptimizationType) -> int:
        """Estimate implementation time for optimization type"""

        time_estimates = {
            OptimizationType.ENERGY_EFFICIENCY: 90,
            OptimizationType.TRAFFIC_FLOW: 60,
            OptimizationType.AIR_QUALITY: 120,
            OptimizationType.RESOURCE_ALLOCATION: 45,
            OptimizationType.ECOSYSTEM_HEALTH: 180,
            OptimizationType.SOCIAL_EQUITY: 240,
            OptimizationType.RESILIENCE_BUILDING: 150
        }

        return time_estimates.get(opt_type, 90)

    def _get_standard_resource_requirements(self, opt_type: OptimizationType) -> Dict[str, Any]:
        """Get standard resource requirements for optimization type"""

        return {
            "technical_teams": 2,
            "budget_allocation": 25000,
            "system_access": "standard"
        }

    def _get_standard_prerequisites(self, opt_type: OptimizationType) -> List[str]:
        """Get standard prerequisites for optimization type"""

        return ["system_assessment", "stakeholder_notification", "backup_verification"]

    async def record_optimization_result(self, action_id: str, status: str,
                                       actual_impact: Dict[str, float],
                                       duration_minutes: int) -> OptimizationResult:
        """Record the result of an optimization action for learning"""

        # Calculate success score
        original_action = next((a for a in self.active_optimizations if a.action_id == action_id), None)

        if original_action:
            # Compare actual vs expected impact
            impact_accuracy = 0.0
            impact_comparisons = 0

            for domain, expected in original_action.expected_impact.items():
                if domain in actual_impact:
                    if expected != 0:
                        accuracy = 1 - abs(actual_impact[domain] - expected) / abs(expected)
                        accuracy = max(0, min(1, accuracy))
                        impact_accuracy += accuracy
                        impact_comparisons += 1

            if impact_comparisons > 0:
                impact_accuracy /= impact_comparisons
            else:
                impact_accuracy = 0.5  # Default if no comparisons possible

            # Calculate overall success score
            status_scores = {"success": 1.0, "partial": 0.6, "failed": 0.0}
            status_score = status_scores.get(status, 0.0)

            success_score = (status_score * 0.6 + impact_accuracy * 0.4)

        else:
            success_score = 0.5  # Default if action not found

        # Create result record
        result = OptimizationResult(
            action_id=action_id,
            status=status,
            actual_impact=actual_impact,
            implementation_duration_minutes=duration_minutes,
            success_score=success_score,
            lessons_learned=self._extract_lessons_learned(action_id, status, actual_impact),
            timestamp=datetime.now()
        )

        # Store result
        self.optimization_history.append(result)

        # Update learning
        await self._update_learning_from_result(result)

        logger.info(f"Recorded optimization result: {action_id} - {status} (success: {success_score:.2f})")

        return result

    def _extract_lessons_learned(self, action_id: str, status: str,
                               actual_impact: Dict[str, float]) -> List[str]:
        """Extract lessons learned from optimization result"""

        lessons = []

        if status == "success":
            lessons.append("Optimization parameters were well-calibrated")
            if any(impact > 20 for impact in actual_impact.values()):
                lessons.append("High-impact optimization achieved - consider similar approaches")
        elif status == "partial":
            lessons.append("Optimization had mixed results - review parameter tuning")
            lessons.append("Consider extending implementation time or resources")
        else:  # failed
            lessons.append("Optimization failed - review prerequisites and assumptions")
            lessons.append("Consider alternative optimization strategies")

        # Impact-specific lessons
        for domain, impact in actual_impact.items():
            if impact < -10:
                lessons.append(f"Negative impact on {domain} - review approach")
            elif impact > 25:
                lessons.append(f"Exceptional positive impact on {domain} - replicate approach")

        return lessons

    async def _update_learning_from_result(self, result: OptimizationResult):
        """Update learning algorithms based on optimization result"""

        # Update strategy parameters based on success
        original_action = next((a for a in self.active_optimizations if a.action_id == result.action_id), None)

        if original_action and result.success_score > self.success_threshold:
            # Successful optimization - reinforce parameters
            opt_type = original_action.optimization_type

            if opt_type in self.strategies:
                strategy = self.strategies[opt_type]

                # Update parameters with learning rate
                for param, value in original_action.parameters.items():
                    if param in strategy["parameters"]:
                        current_value = strategy["parameters"][param]
                        if isinstance(current_value, (int, float)) and isinstance(value, (int, float)):
                            # Move towards successful value
                            new_value = current_value + self.learning_rate * (value - current_value)
                            strategy["parameters"][param] = new_value

        # Clean up old active optimizations
        self.active_optimizations = [
            a for a in self.active_optimizations if a.action_id != result.action_id
        ]

    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization engine status"""

        recent_results = [
            r for r in self.optimization_history
            if r.timestamp > datetime.now() - timedelta(hours=24)
        ]

        success_rate = 0.0
        if recent_results:
            success_rate = sum(1 for r in recent_results if r.status == "success") / len(recent_results)

        avg_success_score = 0.0
        if recent_results:
            avg_success_score = statistics.mean([r.success_score for r in recent_results])

        return {
            "active_optimizations": len(self.active_optimizations),
            "total_optimization_history": len(self.optimization_history),
            "recent_optimizations_24h": len(recent_results),
            "success_rate_24h": success_rate,
            "average_success_score": avg_success_score,
            "learning_parameters": {
                "learning_rate": self.learning_rate,
                "adaptation_threshold": self.adaptation_threshold,
                "system_memory_size": len(self.system_state_memory)
            },
            "optimization_types_available": [t.value for t in OptimizationType]
        }