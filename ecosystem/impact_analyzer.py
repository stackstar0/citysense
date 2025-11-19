"""
Impact Analyzer - Comprehensive climate-economy-ecology impact modeling
Analyzes interconnections and ripple effects across urban systems
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

class ImpactDomain(Enum):
    CLIMATE = "climate"
    ECONOMY = "economy"
    ECOLOGY = "ecology"
    SOCIAL = "social"
    INFRASTRUCTURE = "infrastructure"

class ImpactType(Enum):
    DIRECT = "direct"
    INDIRECT = "indirect"
    CASCADING = "cascading"
    FEEDBACK = "feedback"

@dataclass
class ImpactAssessment:
    """Comprehensive impact assessment result"""
    assessment_id: str
    trigger_event: str
    trigger_domain: ImpactDomain
    timestamp: datetime

    # Impact scores by domain (0-100, negative for harmful impacts)
    climate_impact: float
    economic_impact: float
    ecological_impact: float
    social_impact: float
    infrastructure_impact: float

    # Impact details
    impact_chain: List[Dict[str, Any]]  # Chain of cause-effect relationships
    confidence_score: float  # 0-1
    time_horizon_hours: int
    affected_areas: List[str]

    # Recommendations
    mitigation_strategies: List[str]
    adaptation_recommendations: List[str]
    monitoring_priorities: List[str]

    metadata: Dict[str, Any]

@dataclass
class SystemInterconnection:
    """Represents interconnection between urban systems"""
    system_pair: Tuple[str, str]
    connection_type: str  # "dependency", "feedback", "competition", "synergy"
    strength: float  # 0-1
    direction: str  # "bidirectional", "unidirectional"
    delay_hours: int  # Time delay for impact propagation
    description: str

class ImpactAnalyzer:
    """
    Advanced impact analysis system for urban ecosystem modeling
    Models complex interactions between climate, economy, and ecology
    """

    def __init__(self):
        self.system_interconnections: List[SystemInterconnection] = []
        self.impact_history: List[ImpactAssessment] = []

        # Impact propagation weights (how impacts spread between domains)
        self.propagation_matrix = self._initialize_propagation_matrix()

        # Initialize system interconnections
        self._initialize_system_interconnections()

        # Impact thresholds
        self.minor_impact_threshold = 10
        self.major_impact_threshold = 30
        self.critical_impact_threshold = 70

        logger.info("🌍 Impact analyzer initialized with ecosystem modeling")

    def _initialize_propagation_matrix(self) -> Dict[ImpactDomain, Dict[ImpactDomain, float]]:
        """Initialize impact propagation weights between domains"""

        # How much impact from source domain affects target domain (0-1)
        matrix = {
            ImpactDomain.CLIMATE: {
                ImpactDomain.CLIMATE: 1.0,
                ImpactDomain.ECONOMY: 0.6,  # Climate affects economy moderately
                ImpactDomain.ECOLOGY: 0.9,  # Climate strongly affects ecology
                ImpactDomain.SOCIAL: 0.4,   # Climate affects social systems
                ImpactDomain.INFRASTRUCTURE: 0.7  # Climate affects infrastructure
            },
            ImpactDomain.ECONOMY: {
                ImpactDomain.CLIMATE: 0.5,  # Economic activity affects climate
                ImpactDomain.ECONOMY: 1.0,
                ImpactDomain.ECOLOGY: 0.7,  # Economy affects ecology strongly
                ImpactDomain.SOCIAL: 0.8,   # Economy strongly affects social systems
                ImpactDomain.INFRASTRUCTURE: 0.6  # Economy affects infrastructure
            },
            ImpactDomain.ECOLOGY: {
                ImpactDomain.CLIMATE: 0.3,  # Ecology affects climate (carbon sequestration, etc.)
                ImpactDomain.ECONOMY: 0.4,  # Ecosystem services affect economy
                ImpactDomain.ECOLOGY: 1.0,
                ImpactDomain.SOCIAL: 0.5,   # Ecological health affects social wellbeing
                ImpactDomain.INFRASTRUCTURE: 0.2  # Ecology has limited direct infrastructure impact
            },
            ImpactDomain.SOCIAL: {
                ImpactDomain.CLIMATE: 0.3,  # Social behavior affects climate
                ImpactDomain.ECONOMY: 0.7,  # Social systems strongly affect economy
                ImpactDomain.ECOLOGY: 0.4,  # Social systems affect ecology
                ImpactDomain.SOCIAL: 1.0,
                ImpactDomain.INFRASTRUCTURE: 0.5  # Social demand affects infrastructure
            },
            ImpactDomain.INFRASTRUCTURE: {
                ImpactDomain.CLIMATE: 0.4,  # Infrastructure affects climate (emissions, etc.)
                ImpactDomain.ECONOMY: 0.8,  # Infrastructure strongly enables economy
                ImpactDomain.ECOLOGY: 0.3,  # Infrastructure affects ecology
                ImpactDomain.SOCIAL: 0.6,   # Infrastructure affects social systems
                ImpactDomain.INFRASTRUCTURE: 1.0
            }
        }

        return matrix

    def _initialize_system_interconnections(self):
        """Initialize known interconnections between urban systems"""

        interconnections = [
            # Climate-Economy interconnections
            SystemInterconnection(
                system_pair=("climate", "energy_consumption"),
                connection_type="feedback",
                strength=0.8,
                direction="bidirectional",
                delay_hours=2,
                description="Energy consumption affects climate through emissions; climate affects energy demand"
            ),

            SystemInterconnection(
                system_pair=("air_quality", "economic_activity"),
                connection_type="competition",
                strength=0.6,
                direction="bidirectional",
                delay_hours=6,
                description="Industrial activity reduces air quality; poor air quality reduces economic productivity"
            ),

            # Economy-Ecology interconnections
            SystemInterconnection(
                system_pair=("economic_growth", "resource_consumption"),
                connection_type="dependency",
                strength=0.9,
                direction="unidirectional",
                delay_hours=1,
                description="Economic growth depends on resource availability and consumption"
            ),

            SystemInterconnection(
                system_pair=("waste_generation", "ecological_health"),
                connection_type="feedback",
                strength=0.7,
                direction="bidirectional",
                delay_hours=12,
                description="Economic activity generates waste affecting ecology; poor ecology affects economic costs"
            ),

            # Climate-Ecology interconnections
            SystemInterconnection(
                system_pair=("temperature", "biodiversity"),
                connection_type="dependency",
                strength=0.8,
                direction="bidirectional",
                delay_hours=24,
                description="Temperature changes affect biodiversity; biodiversity affects local climate regulation"
            ),

            SystemInterconnection(
                system_pair=("precipitation", "water_quality"),
                connection_type="dependency",
                strength=0.9,
                direction="unidirectional",
                delay_hours=4,
                description="Precipitation patterns strongly affect water quality and availability"
            ),

            # Social system interconnections
            SystemInterconnection(
                system_pair=("air_quality", "public_health"),
                connection_type="dependency",
                strength=0.9,
                direction="unidirectional",
                delay_hours=8,
                description="Air quality directly affects public health outcomes"
            ),

            SystemInterconnection(
                system_pair=("green_space", "social_wellbeing"),
                connection_type="synergy",
                strength=0.7,
                direction="unidirectional",
                delay_hours=2,
                description="Green spaces enhance social wellbeing and community health"
            ),

            # Infrastructure interconnections
            SystemInterconnection(
                system_pair=("traffic_congestion", "air_quality"),
                connection_type="feedback",
                strength=0.8,
                direction="unidirectional",
                delay_hours=1,
                description="Traffic congestion directly affects local air quality"
            ),

            SystemInterconnection(
                system_pair=("energy_grid", "economic_activity"),
                connection_type="dependency",
                strength=0.9,
                direction="bidirectional",
                delay_hours=0,
                description="Economic activity depends on energy infrastructure; energy demand affects grid"
            )
        ]

        self.system_interconnections = interconnections
        logger.info(f"Initialized {len(interconnections)} system interconnections")

    async def analyze_impact(self, trigger_event: str, trigger_domain: ImpactDomain,
                           trigger_magnitude: float, current_state: Dict[str, Any],
                           time_horizon_hours: int = 24) -> ImpactAssessment:
        """
        Comprehensive impact analysis for a trigger event
        Models cascading effects across all urban systems
        """

        logger.info(f"Analyzing impact: {trigger_event} in {trigger_domain.value} domain")

        try:
            # Initialize impact scores
            impact_scores = {
                ImpactDomain.CLIMATE: 0.0,
                ImpactDomain.ECONOMY: 0.0,
                ImpactDomain.ECOLOGY: 0.0,
                ImpactDomain.SOCIAL: 0.0,
                ImpactDomain.INFRASTRUCTURE: 0.0
            }

            # Set initial impact in trigger domain
            impact_scores[trigger_domain] = trigger_magnitude

            # Calculate cascading impacts
            impact_chain = []
            total_iterations = 5  # Number of propagation iterations

            for iteration in range(total_iterations):
                new_impacts = impact_scores.copy()

                # Propagate impacts through interconnection matrix
                for source_domain, source_impact in impact_scores.items():
                    if abs(source_impact) < 1:  # Skip negligible impacts
                        continue

                    for target_domain, propagation_weight in self.propagation_matrix[source_domain].items():
                        if source_domain != target_domain:
                            # Calculate propagated impact with decay
                            decay_factor = 0.8 ** iteration  # Impacts decay over iterations
                            propagated_impact = source_impact * propagation_weight * decay_factor

                            # Apply system-specific modifiers
                            modifier = self._get_system_modifier(
                                source_domain, target_domain, current_state
                            )
                            propagated_impact *= modifier

                            new_impacts[target_domain] += propagated_impact

                            # Record impact chain
                            if abs(propagated_impact) > 5:  # Record significant impacts
                                impact_chain.append({
                                    "iteration": iteration + 1,
                                    "source_domain": source_domain.value,
                                    "target_domain": target_domain.value,
                                    "impact_magnitude": propagated_impact,
                                    "propagation_weight": propagation_weight,
                                    "system_modifier": modifier
                                })

                impact_scores = new_impacts

            # Calculate system-specific impacts using interconnections
            interconnection_impacts = await self._analyze_interconnection_impacts(
                trigger_event, trigger_domain, trigger_magnitude, current_state
            )

            # Merge interconnection impacts
            for domain, additional_impact in interconnection_impacts.items():
                impact_scores[domain] += additional_impact

            # Calculate confidence score
            confidence = self._calculate_confidence_score(
                trigger_event, trigger_domain, current_state, impact_chain
            )

            # Determine affected areas
            affected_areas = self._determine_affected_areas(impact_scores, current_state)

            # Generate recommendations
            mitigation_strategies = self._generate_mitigation_strategies(
                trigger_event, trigger_domain, impact_scores
            )
            adaptation_recommendations = self._generate_adaptation_recommendations(
                impact_scores, time_horizon_hours
            )
            monitoring_priorities = self._generate_monitoring_priorities(
                impact_scores, impact_chain
            )

            # Create impact assessment
            assessment = ImpactAssessment(
                assessment_id=f"impact_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{trigger_domain.value}",
                trigger_event=trigger_event,
                trigger_domain=trigger_domain,
                timestamp=datetime.now(),
                climate_impact=impact_scores[ImpactDomain.CLIMATE],
                economic_impact=impact_scores[ImpactDomain.ECONOMY],
                ecological_impact=impact_scores[ImpactDomain.ECOLOGY],
                social_impact=impact_scores[ImpactDomain.SOCIAL],
                infrastructure_impact=impact_scores[ImpactDomain.INFRASTRUCTURE],
                impact_chain=impact_chain,
                confidence_score=confidence,
                time_horizon_hours=time_horizon_hours,
                affected_areas=affected_areas,
                mitigation_strategies=mitigation_strategies,
                adaptation_recommendations=adaptation_recommendations,
                monitoring_priorities=monitoring_priorities,
                metadata={
                    "trigger_magnitude": trigger_magnitude,
                    "propagation_iterations": total_iterations,
                    "interconnections_analyzed": len(self.system_interconnections),
                    "analysis_timestamp": datetime.now().isoformat()
                }
            )

            # Store in history
            self.impact_history.append(assessment)

            logger.info(f"Impact analysis complete: {len(impact_chain)} cascade effects identified")

            return assessment

        except Exception as e:
            logger.error(f"Error in impact analysis: {e}")
            raise

    def _get_system_modifier(self, source_domain: ImpactDomain, target_domain: ImpactDomain,
                           current_state: Dict[str, Any]) -> float:
        """Calculate system-specific modifiers based on current state"""

        modifier = 1.0

        try:
            # Climate-related modifiers
            if source_domain == ImpactDomain.CLIMATE:
                # Air quality affects climate impact propagation
                air_quality = self._extract_air_quality_score(current_state)
                if air_quality < 50:  # Poor air quality amplifies climate impacts
                    modifier *= 1.2

            # Economic modifiers
            if source_domain == ImpactDomain.ECONOMY:
                # Economic stress affects impact propagation
                economic_activity = self._extract_economic_activity(current_state)
                if economic_activity < 40:  # Economic stress amplifies impacts
                    modifier *= 1.3
                elif economic_activity > 80:  # Strong economy dampens some impacts
                    modifier *= 0.9

            # Ecological modifiers
            if source_domain == ImpactDomain.ECOLOGY:
                # Ecological health affects resilience
                ecological_health = self._extract_ecological_health(current_state)
                if ecological_health > 70:  # Healthy ecology provides resilience
                    modifier *= 0.8
                elif ecological_health < 30:  # Poor ecology amplifies impacts
                    modifier *= 1.4

            # Infrastructure modifiers
            if source_domain == ImpactDomain.INFRASTRUCTURE:
                # Infrastructure efficiency affects impact propagation
                energy_efficiency = self._extract_energy_efficiency(current_state)
                if energy_efficiency > 75:  # Efficient infrastructure dampens impacts
                    modifier *= 0.9
                elif energy_efficiency < 40:  # Inefficient infrastructure amplifies impacts
                    modifier *= 1.2

        except Exception as e:
            logger.debug(f"Error calculating system modifier: {e}")

        return max(0.1, min(2.0, modifier))  # Clamp between 0.1 and 2.0

    async def _analyze_interconnection_impacts(self, trigger_event: str, trigger_domain: ImpactDomain,
                                             trigger_magnitude: float, current_state: Dict[str, Any]) -> Dict[ImpactDomain, float]:
        """Analyze impacts through specific system interconnections"""

        interconnection_impacts = {domain: 0.0 for domain in ImpactDomain}

        # Find relevant interconnections
        relevant_interconnections = [
            ic for ic in self.system_interconnections
            if self._is_interconnection_relevant(ic, trigger_event, trigger_domain)
        ]

        for interconnection in relevant_interconnections:
            try:
                # Calculate impact through this interconnection
                impact_magnitude = trigger_magnitude * interconnection.strength

                # Apply time decay if analyzing future impacts
                if interconnection.delay_hours > 0:
                    decay_factor = 0.95 ** interconnection.delay_hours
                    impact_magnitude *= decay_factor

                # Determine target domain for this interconnection
                target_domains = self._get_interconnection_target_domains(
                    interconnection, trigger_event
                )

                for target_domain in target_domains:
                    interconnection_impacts[target_domain] += impact_magnitude

            except Exception as e:
                logger.debug(f"Error analyzing interconnection {interconnection.system_pair}: {e}")

        return interconnection_impacts

    def _is_interconnection_relevant(self, interconnection: SystemInterconnection,
                                   trigger_event: str, trigger_domain: ImpactDomain) -> bool:
        """Check if an interconnection is relevant to the trigger event"""

        # Map trigger domains to system keywords
        domain_keywords = {
            ImpactDomain.CLIMATE: ["climate", "temperature", "air_quality", "weather", "emissions"],
            ImpactDomain.ECONOMY: ["economic", "energy_consumption", "activity", "growth"],
            ImpactDomain.ECOLOGY: ["ecological", "biodiversity", "water_quality", "green_space", "waste"],
            ImpactDomain.SOCIAL: ["social", "health", "wellbeing", "public"],
            ImpactDomain.INFRASTRUCTURE: ["traffic", "energy_grid", "infrastructure"]
        }

        trigger_keywords = domain_keywords.get(trigger_domain, [])

        # Check if interconnection involves systems relevant to the trigger
        system1, system2 = interconnection.system_pair

        for keyword in trigger_keywords:
            if keyword in system1.lower() or keyword in system2.lower():
                return True

        # Check trigger event text
        if any(keyword in trigger_event.lower() for keyword in [system1, system2]):
            return True

        return False

    def _get_interconnection_target_domains(self, interconnection: SystemInterconnection,
                                          trigger_event: str) -> List[ImpactDomain]:
        """Determine which domains are affected by an interconnection"""

        target_domains = []
        system1, system2 = interconnection.system_pair

        # Map systems to domains
        system_domain_map = {
            "climate": [ImpactDomain.CLIMATE],
            "temperature": [ImpactDomain.CLIMATE],
            "air_quality": [ImpactDomain.CLIMATE, ImpactDomain.SOCIAL],
            "energy_consumption": [ImpactDomain.ECONOMY, ImpactDomain.CLIMATE],
            "economic_activity": [ImpactDomain.ECONOMY],
            "economic_growth": [ImpactDomain.ECONOMY],
            "resource_consumption": [ImpactDomain.ECONOMY, ImpactDomain.ECOLOGY],
            "waste_generation": [ImpactDomain.ECOLOGY],
            "ecological_health": [ImpactDomain.ECOLOGY],
            "biodiversity": [ImpactDomain.ECOLOGY],
            "water_quality": [ImpactDomain.ECOLOGY, ImpactDomain.SOCIAL],
            "green_space": [ImpactDomain.ECOLOGY, ImpactDomain.SOCIAL],
            "public_health": [ImpactDomain.SOCIAL],
            "social_wellbeing": [ImpactDomain.SOCIAL],
            "traffic_congestion": [ImpactDomain.INFRASTRUCTURE, ImpactDomain.CLIMATE],
            "energy_grid": [ImpactDomain.INFRASTRUCTURE]
        }

        # Get domains for both systems in the interconnection
        for system in [system1, system2]:
            domains = system_domain_map.get(system, [])
            target_domains.extend(domains)

        return list(set(target_domains))  # Remove duplicates

    def _calculate_confidence_score(self, trigger_event: str, trigger_domain: ImpactDomain,
                                  current_state: Dict[str, Any], impact_chain: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the impact assessment"""

        confidence_factors = []

        # Data quality factor
        data_quality = self._assess_data_quality(current_state)
        confidence_factors.append(data_quality * 0.3)

        # Model certainty factor (based on number of well-understood interconnections)
        known_interconnections = len([ic for ic in self.system_interconnections
                                    if ic.strength > 0.6])
        model_certainty = min(1.0, known_interconnections / 10)
        confidence_factors.append(model_certainty * 0.3)

        # Impact chain consistency factor
        if impact_chain:
            chain_consistency = 1.0 - (len(impact_chain) * 0.05)  # Longer chains less certain
            chain_consistency = max(0.3, chain_consistency)
        else:
            chain_consistency = 0.8
        confidence_factors.append(chain_consistency * 0.2)

        # Historical validation factor (simplified)
        historical_factor = 0.7  # Assume moderate historical validation
        confidence_factors.append(historical_factor * 0.2)

        return sum(confidence_factors)

    def _assess_data_quality(self, current_state: Dict[str, Any]) -> float:
        """Assess quality of current state data"""

        quality_score = 0.0
        quality_checks = 0

        # Check for presence of key data categories
        required_categories = ['air_quality_sensors', 'traffic_sensors', 'energy_sensors']

        for category in required_categories:
            if category in current_state:
                sensors = current_state[category]
                if isinstance(sensors, list) and len(sensors) > 0:
                    # Check data completeness
                    complete_sensors = 0
                    for sensor in sensors:
                        if isinstance(sensor, dict) and len(sensor) > 3:  # Has multiple measurements
                            complete_sensors += 1

                    if complete_sensors > 0:
                        quality_score += complete_sensors / len(sensors)
                        quality_checks += 1

        return quality_score / max(1, quality_checks)

    def _determine_affected_areas(self, impact_scores: Dict[ImpactDomain, float],
                                current_state: Dict[str, Any]) -> List[str]:
        """Determine which geographic/functional areas are affected"""

        affected_areas = []

        # Map impact types to affected areas
        if abs(impact_scores[ImpactDomain.CLIMATE]) > self.minor_impact_threshold:
            affected_areas.extend(["citywide", "industrial_areas", "transportation_corridors"])

        if abs(impact_scores[ImpactDomain.ECONOMY]) > self.minor_impact_threshold:
            affected_areas.extend(["business_district", "commercial_areas", "industrial_zones"])

        if abs(impact_scores[ImpactDomain.ECOLOGY]) > self.minor_impact_threshold:
            affected_areas.extend(["parks_and_green_spaces", "waterways", "natural_reserves"])

        if abs(impact_scores[ImpactDomain.SOCIAL]) > self.minor_impact_threshold:
            affected_areas.extend(["residential_areas", "schools", "healthcare_facilities"])

        if abs(impact_scores[ImpactDomain.INFRASTRUCTURE]) > self.minor_impact_threshold:
            affected_areas.extend(["transportation_network", "energy_grid", "water_systems"])

        return list(set(affected_areas))  # Remove duplicates

    def _generate_mitigation_strategies(self, trigger_event: str, trigger_domain: ImpactDomain,
                                      impact_scores: Dict[ImpactDomain, float]) -> List[str]:
        """Generate mitigation strategies based on impact analysis"""

        strategies = []

        # Climate mitigation
        if abs(impact_scores[ImpactDomain.CLIMATE]) > self.minor_impact_threshold:
            strategies.extend([
                "Activate air quality improvement systems",
                "Implement traffic reduction measures",
                "Increase renewable energy utilization",
                "Deploy urban cooling strategies"
            ])

        # Economic mitigation
        if abs(impact_scores[ImpactDomain.ECONOMY]) > self.minor_impact_threshold:
            strategies.extend([
                "Implement economic stimulus measures",
                "Support affected businesses and industries",
                "Optimize resource allocation efficiency",
                "Diversify economic activities"
            ])

        # Ecological mitigation
        if abs(impact_scores[ImpactDomain.ECOLOGY]) > self.minor_impact_threshold:
            strategies.extend([
                "Enhance ecosystem restoration programs",
                "Implement biodiversity protection measures",
                "Improve waste management systems",
                "Strengthen water quality protection"
            ])

        # Social mitigation
        if abs(impact_scores[ImpactDomain.SOCIAL]) > self.minor_impact_threshold:
            strategies.extend([
                "Enhance public health monitoring",
                "Improve access to essential services",
                "Strengthen community resilience programs",
                "Address social equity concerns"
            ])

        # Infrastructure mitigation
        if abs(impact_scores[ImpactDomain.INFRASTRUCTURE]) > self.minor_impact_threshold:
            strategies.extend([
                "Upgrade critical infrastructure systems",
                "Implement redundancy measures",
                "Enhance maintenance and monitoring",
                "Optimize system efficiency"
            ])

        return strategies[:10]  # Limit to top 10 strategies

    def _generate_adaptation_recommendations(self, impact_scores: Dict[ImpactDomain, float],
                                           time_horizon_hours: int) -> List[str]:
        """Generate adaptation recommendations for long-term resilience"""

        recommendations = []

        total_impact = sum(abs(score) for score in impact_scores.values())

        if total_impact > self.critical_impact_threshold:
            recommendations.extend([
                "Develop comprehensive crisis response protocols",
                "Establish emergency resource reserves",
                "Create alternative system pathways",
                "Enhance inter-system coordination"
            ])
        elif total_impact > self.major_impact_threshold:
            recommendations.extend([
                "Strengthen system monitoring and early warning",
                "Implement adaptive capacity building",
                "Develop scenario-based response plans",
                "Enhance stakeholder coordination"
            ])
        else:
            recommendations.extend([
                "Monitor system performance trends",
                "Implement preventive maintenance",
                "Build knowledge and capacity",
                "Strengthen routine coordination"
            ])

        # Time-horizon specific recommendations
        if time_horizon_hours > 168:  # More than a week
            recommendations.extend([
                "Develop long-term adaptation strategies",
                "Invest in resilient infrastructure",
                "Build institutional capacity",
                "Foster community engagement"
            ])

        return recommendations

    def _generate_monitoring_priorities(self, impact_scores: Dict[ImpactDomain, float],
                                      impact_chain: List[Dict[str, Any]]) -> List[str]:
        """Generate monitoring priorities based on impact analysis"""

        priorities = []

        # Prioritize domains with highest impacts
        sorted_impacts = sorted(impact_scores.items(), key=lambda x: abs(x[1]), reverse=True)

        for domain, impact in sorted_impacts[:3]:  # Top 3 impacted domains
            if abs(impact) > self.minor_impact_threshold:
                domain_priorities = {
                    ImpactDomain.CLIMATE: [
                        "Air quality monitoring",
                        "Temperature and humidity tracking",
                        "Emissions monitoring"
                    ],
                    ImpactDomain.ECONOMY: [
                        "Economic activity indicators",
                        "Resource consumption patterns",
                        "Business performance metrics"
                    ],
                    ImpactDomain.ECOLOGY: [
                        "Ecosystem health indicators",
                        "Biodiversity monitoring",
                        "Water quality assessment"
                    ],
                    ImpactDomain.SOCIAL: [
                        "Public health indicators",
                        "Social wellbeing metrics",
                        "Community engagement levels"
                    ],
                    ImpactDomain.INFRASTRUCTURE: [
                        "System performance monitoring",
                        "Maintenance status tracking",
                        "Capacity utilization metrics"
                    ]
                }

                priorities.extend(domain_priorities.get(domain, []))

        # Add priorities for interconnected systems
        if impact_chain:
            interconnected_systems = set()
            for impact in impact_chain:
                interconnected_systems.add(impact['source_domain'])
                interconnected_systems.add(impact['target_domain'])

            priorities.append(f"Monitor interconnections between: {', '.join(interconnected_systems)}")

        return priorities[:8]  # Limit to top 8 priorities

    def _extract_air_quality_score(self, current_state: Dict[str, Any]) -> float:
        """Extract air quality score from current state"""
        try:
            if 'air_quality' in current_state:
                return current_state['air_quality'].get('air_quality_index', 50)
            return 50  # Default neutral value
        except:
            return 50

    def _extract_economic_activity(self, current_state: Dict[str, Any]) -> float:
        """Extract economic activity level from current state"""
        try:
            if 'energy_summary' in current_state:
                consumption = current_state['energy_summary'].get('total_consumption_mw', 400)
                # Normalize to 0-100 scale
                return min(100, (consumption / 600) * 100)
            return 50
        except:
            return 50

    def _extract_ecological_health(self, current_state: Dict[str, Any]) -> float:
        """Extract ecological health score from current state"""
        try:
            # Calculate from water quality and air quality
            air_quality = self._extract_air_quality_score(current_state)

            # Simplified ecological health calculation
            ecological_health = air_quality * 0.7 + 30  # Base score + air quality factor
            return min(100, max(0, ecological_health))
        except:
            return 50

    def _extract_energy_efficiency(self, current_state: Dict[str, Any]) -> float:
        """Extract energy efficiency from current state"""
        try:
            if 'energy_summary' in current_state:
                renewable_ratio = current_state['energy_summary'].get('renewable_ratio', 0.3)
                grid_efficiency = current_state['energy_summary'].get('grid_efficiency', 0.85)
                return (renewable_ratio * 50 + grid_efficiency * 50)
            return 60
        except:
            return 60

    async def get_impact_summary(self) -> Dict[str, Any]:
        """Get summary of recent impact assessments"""

        recent_assessments = [
            a for a in self.impact_history
            if a.timestamp > datetime.now() - timedelta(hours=24)
        ]

        if not recent_assessments:
            return {
                "total_assessments": 0,
                "recent_assessments": 0,
                "average_confidence": 0,
                "most_impacted_domains": []
            }

        # Calculate statistics
        total_climate_impact = sum(abs(a.climate_impact) for a in recent_assessments)
        total_economic_impact = sum(abs(a.economic_impact) for a in recent_assessments)
        total_ecological_impact = sum(abs(a.ecological_impact) for a in recent_assessments)
        total_social_impact = sum(abs(a.social_impact) for a in recent_assessments)
        total_infrastructure_impact = sum(abs(a.infrastructure_impact) for a in recent_assessments)

        domain_impacts = {
            "climate": total_climate_impact,
            "economy": total_economic_impact,
            "ecology": total_ecological_impact,
            "social": total_social_impact,
            "infrastructure": total_infrastructure_impact
        }

        most_impacted = sorted(domain_impacts.items(), key=lambda x: x[1], reverse=True)

        return {
            "total_assessments": len(self.impact_history),
            "recent_assessments": len(recent_assessments),
            "average_confidence": statistics.mean([a.confidence_score for a in recent_assessments]),
            "most_impacted_domains": [{"domain": domain, "total_impact": impact} for domain, impact in most_impacted[:3]],
            "critical_assessments": len([a for a in recent_assessments if any(abs(getattr(a, f"{d.value}_impact")) > self.critical_impact_threshold for d in ImpactDomain)]),
            "system_interconnections": len(self.system_interconnections)
        }