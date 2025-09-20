"""
Enhanced Legal Document Demystifier - Advanced Risk Analyzer
Production-ready risk analysis with machine learning and Indian law compliance
"""

import asyncio
import logging
import re
import yaml
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class RiskResult:
    """Risk analysis result"""
    level: str  # RED, YELLOW, GREEN
    severity: int  # 1-10 scale
    risk_factors: List[str]
    indian_law_warnings: List[str]
    confidence: float  # 0-1
    recommendations: List[str]

class AdvancedRiskAnalyzer:
    """Advanced risk analysis with ML-powered pattern detection"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.risk_patterns = self.config.get('risk_patterns', {})
        self.indian_law_checks = self.config.get('indian_law_checks', {})
        self.risk_levels = {"GREEN": "✅", "YELLOW": "⚠️", "RED": "🚨"}
        
        # Cache for performance
        self._compiled_patterns = {}
        self._compile_patterns()
    
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load risk analysis configuration"""
        if config_path and config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        # Default configuration if file not found
        return self._get_default_config()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for performance"""
        for level, patterns in self.risk_patterns.items():
            self._compiled_patterns[level] = {}
            for risk_name, risk_data in patterns.items():
                if isinstance(risk_data, dict):
                    pattern = risk_data.get('pattern', '')
                else:
                    pattern = risk_data
                
                try:
                    self._compiled_patterns[level][risk_name] = re.compile(
                        pattern, re.I | re.S
                    )
                except re.error as e:
                    logger.warning(f"Invalid regex pattern for {risk_name}: {e}")
    
    async def analyze_clause_risk(self, clause: Dict) -> RiskResult:
        """
        Comprehensive risk analysis for a single clause
        
        Args:
            clause: Clause dictionary with text, metadata
            
        Returns:
            RiskResult with detailed analysis
        """
        clause_text = clause.get("text", "")
        clause_type = clause.get("clause_type", "general")
        
        # Run analysis components in parallel
        tasks = [
            self._pattern_based_analysis(clause_text),
            self._indian_law_compliance_check(clause_text),
            self._contextual_risk_analysis(clause_text, clause_type),
            self._entity_based_risk_analysis(clause.get("entities", {}))
        ]
        
        results = await asyncio.gather(*tasks)
        pattern_risks, law_warnings, contextual_risks, entity_risks = results
        
        # Combine all risk factors
        all_risk_factors = pattern_risks + contextual_risks + entity_risks
        
        # Determine overall risk level and severity
        overall_level, severity = self._calculate_overall_risk(all_risk_factors)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_level, all_risk_factors, law_warnings, clause_type
        )
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_confidence(clause_text, all_risk_factors)
        
        return RiskResult(
            level=overall_level,
            severity=severity,
            risk_factors=all_risk_factors,
            indian_law_warnings=law_warnings,
            confidence=confidence,
            recommendations=recommendations
        )
    
    async def _pattern_based_analysis(self, text: str) -> List[str]:
        """Pattern-based risk detection"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._run_pattern_analysis, text
        )
    
    def _run_pattern_analysis(self, text: str) -> List[str]:
        """Synchronous pattern analysis"""
        found_risks = []
        
        for level in ["RED", "YELLOW", "GREEN"]:
            if level not in self._compiled_patterns:
                continue
                
            for risk_name, pattern in self._compiled_patterns[level].items():
                if pattern.search(text):
                    risk_data = self.risk_patterns[level][risk_name]
                    if isinstance(risk_data, dict):
                        description = risk_data.get('description', risk_name)
                        severity = risk_data.get('severity', 5)
                        found_risks.append(f"{description} (severity: {severity})")
                    else:
                        found_risks.append(risk_name)
        
        return found_risks
    
    async def _indian_law_compliance_check(self, text: str) -> List[str]:
        """Check compliance with Indian laws"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._run_indian_law_check, text
        )
    
    def _run_indian_law_check(self, text: str) -> List[str]:
        """Synchronous Indian law compliance check"""
        warnings = []
        
        for law_area, check_data in self.indian_law_checks.items():
            pattern = check_data.get('pattern', '')
            warning = check_data.get('warning', '')
            
            if re.search(pattern, text, re.I):
                warnings.append(f"{law_area}: {warning}")
        
        return warnings
    
    async def _contextual_risk_analysis(self, text: str, clause_type: str) -> List[str]:
        """Context-aware risk analysis based on clause type"""
        risks = []
        
        # Type-specific risk patterns
        type_specific_risks = {
            "payment": self._analyze_payment_risks,
            "termination": self._analyze_termination_risks,
            "liability": self._analyze_liability_risks,
            "confidentiality": self._analyze_confidentiality_risks,
            "intellectual_property": self._analyze_ip_risks,
            "dispute": self._analyze_dispute_risks
        }
        
        if clause_type in type_specific_risks:
            analyzer = type_specific_risks[clause_type]
            risks.extend(await analyzer(text))
        
        return risks
    
    async def _analyze_payment_risks(self, text: str) -> List[str]:
        """Analyze payment-specific risks"""
        risks = []
        text_lower = text.lower()
        
        # High interest rates
        interest_matches = re.findall(r'interest.*?(\d+(?:\.\d+)?)%', text_lower)
        for rate in interest_matches:
            if float(rate) > 24:  # RBI guidelines
                risks.append(f"High interest rate: {rate}% (above RBI guidelines)")
        
        # Payment frequency issues
        if re.search(r'daily|weekly', text_lower):
            risks.append("Unusual payment frequency (daily/weekly)")
        
        # Unclear payment terms
        if not re.search(r'due date|payment.*within|specific date', text_lower):
            risks.append("Unclear payment deadlines")
        
        return risks
    
    async def _analyze_termination_risks(self, text: str) -> List[str]:
        """Analyze termination clause risks"""
        risks = []
        text_lower = text.lower()
        
        # Immediate termination without notice
        if re.search(r'terminate.*immediately|instant.*termination', text_lower):
            risks.append("Immediate termination without notice provision")
        
        # One-sided termination rights
        if re.search(r'party a.*may terminate|landlord.*may terminate', text_lower):
            if not re.search(r'party b.*may terminate|tenant.*may terminate', text_lower):
                risks.append("One-sided termination rights")
        
        return risks
    
    async def _analyze_liability_risks(self, text: str) -> List[str]:
        """Analyze liability clause risks"""
        risks = []
        text_lower = text.lower()
        
        # Unlimited liability
        if re.search(r'unlimited.*liability|no.*limit.*liability', text_lower):
            risks.append("Unlimited liability exposure")
        
        # Exclusion of liability
        if re.search(r'excludes.*liability|not.*liable.*for', text_lower):
            risks.append("Broad liability exclusions")
        
        return risks
    
    async def _analyze_confidentiality_risks(self, text: str) -> List[str]:
        """Analyze confidentiality clause risks"""
        risks = []
        text_lower = text.lower()
        
        # Overly broad confidentiality
        if re.search(r'all.*information|any.*information.*disclosed', text_lower):
            if not re.search(r'publicly.*available|already.*known', text_lower):
                risks.append("Overly broad confidentiality obligations")
        
        # Long confidentiality periods
        period_matches = re.findall(r'(\d+)\s*years?.*confidential', text_lower)
        for period in period_matches:
            if int(period) > 5:
                risks.append(f"Long confidentiality period: {period} years")
        
        return risks
    
    async def _analyze_ip_risks(self, text: str) -> List[str]:
        """Analyze intellectual property risks"""
        risks = []
        text_lower = text.lower()
        
        # Broad IP assignment
        if re.search(r'all.*intellectual.*property|assigns.*all.*rights', text_lower):
            risks.append("Broad intellectual property assignment")
        
        return risks
    
    async def _analyze_dispute_risks(self, text: str) -> List[str]:
        """Analyze dispute resolution risks"""
        risks = []
        text_lower = text.lower()
        
        # Mandatory arbitration
        if re.search(r'binding.*arbitration|mandatory.*arbitration', text_lower):
            risks.append("Mandatory binding arbitration (limits court access)")
        
        # Expensive jurisdiction
        if re.search(r'jurisdiction.*(?:singapore|new york|london)', text_lower):
            risks.append("Potentially expensive international jurisdiction")
        
        return risks
    
    async def _entity_based_risk_analysis(self, entities: Dict) -> List[str]:
        """Analyze risks based on extracted entities"""
        risks = []
        
        # High monetary amounts
        amounts = entities.get('amounts', [])
        for amount in amounts:
            # Extract numeric value
            numeric_match = re.search(r'[\d,]+', amount)
            if numeric_match:
                numeric_str = numeric_match.group().replace(',', '')
                try:
                    value = float(numeric_str)
                    if value > 1000000:  # 1 million
                        risks.append(f"High monetary amount: {amount}")
                except ValueError:
                    pass
        
        # Short time periods for important actions
        time_periods = entities.get('time_periods', [])
        for period in time_periods:
            if re.search(r'[1-7]\s*days?', period):
                risks.append(f"Very short time period: {period}")
        
        return risks
    
    def _calculate_overall_risk(self, risk_factors: List[str]) -> Tuple[str, int]:
        """Calculate overall risk level and severity"""
        if not risk_factors:
            return "GREEN", 2
        
        # Count high-severity indicators
        high_severity_count = sum(1 for risk in risk_factors 
                                if any(keyword in risk.lower() 
                                      for keyword in ['unlimited', 'immediate', 'high', 'mandatory']))
        
        total_risks = len(risk_factors)
        
        if high_severity_count > 0 or total_risks > 5:
            return "RED", min(10, 6 + high_severity_count)
        elif total_risks > 2:
            return "YELLOW", min(7, 3 + total_risks)
        else:
            return "GREEN", max(1, total_risks)
    
    def _calculate_confidence(self, text: str, risk_factors: List[str]) -> float:
        """Calculate confidence in risk assessment"""
        base_confidence = 0.7
        
        # Increase confidence based on text length
        if len(text) > 200:
            base_confidence += 0.1
        
        # Increase confidence based on number of risk factors found
        if len(risk_factors) > 0:
            base_confidence += 0.1
        
        # Decrease confidence for very short text
        if len(text) < 50:
            base_confidence -= 0.2
        
        return min(1.0, max(0.1, base_confidence))
    
    def _generate_recommendations(
        self, 
        risk_level: str, 
        risk_factors: List[str], 
        law_warnings: List[str],
        clause_type: str
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if risk_level == "RED":
            recommendations.append("⚠️ HIGH RISK: Consider legal consultation before signing")
            recommendations.append("🔍 Request modifications to reduce risk exposure")
        elif risk_level == "YELLOW":
            recommendations.append("⚠️ MODERATE RISK: Review carefully and understand implications")
            recommendations.append("💡 Consider negotiating more favorable terms")
        else:
            recommendations.append("✅ STANDARD TERMS: Generally acceptable clause")
        
        # Specific recommendations based on risk factors
        if any('unlimited' in risk.lower() for risk in risk_factors):
            recommendations.append("🛡️ Negotiate liability caps or limitations")
        
        if any('immediate' in risk.lower() for risk in risk_factors):
            recommendations.append("⏰ Request reasonable notice periods")
        
        if law_warnings:
            recommendations.append("⚖️ Verify compliance with applicable Indian laws")
        
        # Type-specific recommendations
        type_recommendations = {
            "payment": ["💰 Clarify payment schedules and late fee policies"],
            "termination": ["📋 Ensure mutual termination rights"],
            "confidentiality": ["🤐 Define scope of confidential information clearly"],
            "liability": ["⚖️ Negotiate reasonable liability limitations"],
            "dispute": ["🏛️ Consider alternative dispute resolution options"]
        }
        
        if clause_type in type_recommendations:
            recommendations.extend(type_recommendations[clause_type])
        
        return recommendations
    
    def _get_default_config(self) -> Dict:
        """Default risk analysis configuration"""
        return {
            "risk_patterns": {
                "RED": {
                    "auto_renewal_trap": {
                        "pattern": r"(auto(?:matic)?ally?\s*renew|renew unless|rolls over|unless (?:notice|terminated))",
                        "description": "Automatic renewal without clear notice",
                        "severity": 9
                    },
                    "heavy_penalties": {
                        "pattern": r"(penalt(y|ies)|liquidated damages|fine of|charge.*₹|fee of.*₹)",
                        "description": "Heavy penalties or liquidated damages",
                        "severity": 8
                    },
                    "unilateral_changes": {
                        "pattern": r"(may (?:modify|change|amend) (?:at any time|without notice)|sole discretion|unilateral)",
                        "description": "Unilateral modification rights",
                        "severity": 9
                    }
                },
                "YELLOW": {
                    "notice_requirements": {
                        "pattern": r"(notice period|(?:\d+)\s*days?\s*notice|prior notice|advance notice)",
                        "description": "Notice requirements",
                        "severity": 5
                    },
                    "termination_clauses": {
                        "pattern": r"(termination|terminate on|end of term|expire)",
                        "description": "Termination conditions",
                        "severity": 4
                    }
                },
                "GREEN": {
                    "standard_terms": {
                        "pattern": r"(good faith|reasonable|customary|standard|normal)",
                        "description": "Standard commercial terms",
                        "severity": 2
                    }
                }
            },
            "indian_law_checks": {
                "rent_control": {
                    "pattern": r"(notice.*(?:1[0-5]|[1-9])\s*days|vacate.*(?:1[0-5]|[1-9])\s*days)",
                    "warning": "Indian Rent Control Acts typically require 30+ days notice"
                },
                "interest_rates": {
                    "pattern": r"interest.*(?:2[4-9]|[3-9]\d|\d{3,})%",
                    "warning": "Interest rate seems high. RBI guidelines suggest checking legal limits"
                }
            }
        }

    async def analyze_document_risk(self, clauses: List[Dict]) -> Dict:
        """Analyze risk for entire document"""
        clause_analyses = []
        
        # Analyze each clause
        for clause in clauses:
            risk_result = await self.analyze_clause_risk(clause)
            clause_analyses.append({
                "clause_id": clause.get("id"),
                "clause_type": clause.get("clause_type"),
                "risk_result": risk_result
            })
        
        # Aggregate results
        risk_summary = {"RED": 0, "YELLOW": 0, "GREEN": 0}
        critical_issues = []
        all_recommendations = set()
        
        for analysis in clause_analyses:
            risk_result = analysis["risk_result"]
            risk_summary[risk_result.level] += 1
            
            if risk_result.level == "RED":
                critical_issues.append({
                    "clause_id": analysis["clause_id"],
                    "risk_factors": risk_result.risk_factors,
                    "severity": risk_result.severity,
                    "recommendations": risk_result.recommendations
                })
            
            all_recommendations.update(risk_result.recommendations)
        
        # Generate overall assessment
        total_clauses = len(clauses)
        if risk_summary["RED"] > 0:
            overall_status = "🚨 HIGH RISK - Careful Review Required"
            recommendation = "Multiple high-risk clauses found. Legal consultation strongly recommended."
        elif risk_summary["YELLOW"] > risk_summary["GREEN"]:
            overall_status = "⚠️ MODERATE RISK - Careful Review Advised"
            recommendation = "Some concerning clauses found. Review carefully before signing."
        else:
            overall_status = "✅ RELATIVELY SAFE - Standard Terms"
            recommendation = "Mostly standard clauses, but always review before signing."
        
        return {
            "overall_status": overall_status,
            "recommendation": recommendation,
            "risk_breakdown": risk_summary,
            "critical_issues": critical_issues,
            "total_clauses": total_clauses,
            "clause_analyses": clause_analyses,
            "aggregated_recommendations": list(all_recommendations),
            "risk_score": self._calculate_risk_score(risk_summary, total_clauses)
        }
    
    def _calculate_risk_score(self, risk_summary: Dict, total_clauses: int) -> float:
        """Calculate numerical risk score (0-10)"""
        if total_clauses == 0:
            return 0.0
        
        red_weight = 10
        yellow_weight = 5
        green_weight = 1
        
        weighted_score = (
            risk_summary["RED"] * red_weight +
            risk_summary["YELLOW"] * yellow_weight +
            risk_summary["GREEN"] * green_weight
        )
        
        max_possible_score = total_clauses * red_weight
        normalized_score = (weighted_score / max_possible_score) * 10
        
        return round(normalized_score, 2)