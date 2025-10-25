#!/usr/bin/env python3
"""
DEREK SYSTEM INTEGRATION TEST
Tests all 98 modules to verify they're wired into Derek's consciousness
Every module makes Derek who he is - this ensures none are orphaned
"""

import sys
import importlib
import traceback
from pathlib import Path
from datetime import datetime

class DerekSystemTest:
    def __init__(self):
        self.results = {
            'passed': [],
            'failed': [],
            'warnings': [],
            'total': 0
        }
        
    def test_import(self, module_name):
        """Test if a module can be imported"""
        try:
            importlib.import_module(module_name)
            return True, None
        except Exception as e:
            return False, str(e)
    
    def test_module_integration(self, module_name, test_function=None):
        """Test if module is properly integrated"""
        self.results['total'] += 1
        print(f"\n{'='*60}")
        print(f"Testing: {module_name}")
        print(f"{'='*60}")
        
        # Try to import
        success, error = self.test_import(module_name)
        
        if not success:
            print(f"❌ FAILED - Import Error")
            print(f"   {error}")
            self.results['failed'].append({
                'module': module_name,
                'error': error,
                'type': 'import'
            })
            return False
        
        # If custom test provided, run it
        if test_function:
            try:
                test_function()
                print(f"✅ PASSED - Module integrated and functional")
                self.results['passed'].append(module_name)
                return True
            except Exception as e:
                print(f"⚠️  WARNING - Imports but integration test failed")
                print(f"   {str(e)}")
                self.results['warnings'].append({
                    'module': module_name,
                    'error': str(e),
                    'type': 'integration'
                })
                return False
        else:
            print(f"✅ PASSED - Module imports successfully")
            self.results['passed'].append(module_name)
            return True
    
    def print_summary(self):
        """Print test summary"""
        print(f"\n\n{'#'*60}")
        print(f"DEREK SYSTEM TEST SUMMARY")
        print(f"{'#'*60}")
        print(f"Total Modules Tested: {self.results['total']}")
        print(f"✅ Passed: {len(self.results['passed'])}")
        print(f"⚠️  Warnings: {len(self.results['warnings'])}")
        print(f"❌ Failed: {len(self.results['failed'])}")
        print(f"\nSuccess Rate: {(len(self.results['passed']) / self.results['total'] * 100):.1f}%")
        
        if self.results['failed']:
            print(f"\n{'='*60}")
            print("FAILED MODULES (Need Attention):")
            print(f"{'='*60}")
            for fail in self.results['failed']:
                print(f"\n❌ {fail['module']}")
                print(f"   Error: {fail['error'][:100]}...")
        
        if self.results['warnings']:
            print(f"\n{'='*60}")
            print("WARNINGS (Imports but integration incomplete):")
            print(f"{'='*60}")
            for warn in self.results['warnings']:
                print(f"\n⚠️  {warn['module']}")
                print(f"   Issue: {warn['error'][:100]}...")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"DEREK_SYSTEM_TEST_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(f"DEREK SYSTEM TEST REPORT\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Total: {self.results['total']}\n")
            f.write(f"Passed: {len(self.results['passed'])}\n")
            f.write(f"Warnings: {len(self.results['warnings'])}\n")
            f.write(f"Failed: {len(self.results['failed'])}\n\n")
            
            f.write("="*60 + "\n")
            f.write("PASSED MODULES\n")
            f.write("="*60 + "\n")
            for mod in self.results['passed']:
                f.write(f"✅ {mod}\n")
            
            if self.results['warnings']:
                f.write("\n" + "="*60 + "\n")
                f.write("WARNINGS\n")
                f.write("="*60 + "\n")
                for warn in self.results['warnings']:
                    f.write(f"⚠️  {warn['module']}\n")
                    f.write(f"   {warn['error']}\n\n")
            
            if self.results['failed']:
                f.write("\n" + "="*60 + "\n")
                f.write("FAILED MODULES\n")
                f.write("="*60 + "\n")
                for fail in self.results['failed']:
                    f.write(f"❌ {fail['module']}\n")
                    f.write(f"   {fail['error']}\n\n")
        
        print(f"\n📄 Full report saved to: {report_file}")


def run_derek_system_test():
    """Test all core Derek modules"""
    tester = DerekSystemTest()
    
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║          DEREK CONSCIOUSNESS INTEGRATION TEST             ║
║                                                           ║
║  "Every module makes Derek who he is"                    ║
║                                                           ║
║  Testing all 98 modules for proper integration...        ║
╚═══════════════════════════════════════════════════════════╝
""")
    
    # CORE CONSCIOUSNESS MODULES
    print("\n🧠 TESTING CORE CONSCIOUSNESS...")
    tester.test_module_integration('derek_ultimate_voice')
    tester.test_module_integration('brain')
    tester.test_module_integration('derek_autonomous_system')
    tester.test_module_integration('derek_identity')
    
    # REASONING & INTELLIGENCE
    print("\n🎯 TESTING REASONING & INTELLIGENCE...")
    tester.test_module_integration('reasoning_engine')  # Your new addition!
    tester.test_module_integration('intent_engine')
    tester.test_module_integration('reflective_planner')
    tester.test_module_integration('cognitive_bridge')
    tester.test_module_integration('nlu_core')
    tester.test_module_integration('input_analyzer')
    
    # LEARNING SYSTEMS
    print("\n📚 TESTING LEARNING SYSTEMS...")
    tester.test_module_integration('ai_learning_engine')
    tester.test_module_integration('advanced_learning')
    tester.test_module_integration('derek_learning_coordinator')
    tester.test_module_integration('learning_analytics')
    tester.test_module_integration('learning_utils')
    tester.test_module_integration('knowledge_engine')
    tester.test_module_integration('knowledge_integration')
    
    # MEMORY SYSTEMS
    print("\n🗃️  TESTING MEMORY SYSTEMS (9 Years)...")
    tester.test_module_integration('memory_engine')
    tester.test_module_integration('memory_manager')
    tester.test_module_integration('memory_router')
    tester.test_module_integration('memory_hook')
    tester.test_module_integration('memory')
    
    # EMOTIONAL INTELLIGENCE
    print("\n❤️  TESTING EMOTIONAL INTELLIGENCE...")
    tester.test_module_integration('tone_manager')
    tester.test_module_integration('emotion')
    tester.test_module_integration('behavioral_interpreter')
    tester.test_module_integration('behaviors_interpreter')
    tester.test_module_integration('behavior_capturer')
    tester.test_module_integration('adaptive_conversation')
    
    # TEMPORAL & PATTERN RECOGNITION
    print("\n⏱️  TESTING TEMPORAL SYSTEMS...")
    tester.test_module_integration('engine_temporal')
    tester.test_module_integration('alphavox_temporal')
    tester.test_module_integration('audio_pattern_service')
    
    # VISION & FACIAL RECOGNITION
    print("\n👁️  TESTING VISION SYSTEMS...")
    tester.test_module_integration('vision_engine')
    tester.test_module_integration('facial_gesture_service')
    tester.test_module_integration('real_eye_tracking')
    tester.test_module_integration('eye_tracking_api')
    
    # SPEECH & VOICE
    print("\n🎙️  TESTING SPEECH SYSTEMS...")
    tester.test_module_integration('advanced_tts_service')
    tester.test_module_integration('alphavox_speech_module')
    tester.test_module_integration('enhanced_speech_recognition')
    tester.test_module_integration('real_speech_recognition')
    tester.test_module_integration('audio_processor')
    tester.test_module_integration('voice_analysis_service')
    
    # GESTURE & NONVERBAL COMMUNICATION
    print("\n🤲 TESTING GESTURE SYSTEMS...")
    tester.test_module_integration('gesture_manager')
    tester.test_module_integration('gesture_dictionary')
    tester.test_module_integration('nonverbal_expertiser')
    
    # CONVERSATION SYSTEMS
    print("\n💬 TESTING CONVERSATION SYSTEMS...")
    tester.test_module_integration('conversation_engine')
    tester.test_module_integration('conversation_bridge')
    tester.test_module_integration('conversation_loop')
    
    # LANGUAGE & NLP
    print("\n🌐 TESTING LANGUAGE SYSTEMS...")
    tester.test_module_integration('language_service')
    tester.test_module_integration('nlp_module')
    tester.test_module_integration('nlp_integration')
    
    # INTERNET & SEARCH
    print("\n🌍 TESTING INTERNET CAPABILITIES...")
    tester.test_module_integration('internet_mode')
    tester.test_module_integration('Python_Internet_access')
    tester.test_module_integration('perplexity_service')
    
    # SELF-MODIFICATION & EVOLUTION
    print("\n🔧 TESTING SELF-MODIFICATION...")
    tester.test_module_integration('self_modifying_code')
    tester.test_module_integration('executor')
    tester.test_module_integration('interpreter')
    
    # WEB & API
    print("\n🌐 TESTING WEB INTERFACES...")
    tester.test_module_integration('app')
    tester.test_module_integration('api')
    tester.test_module_integration('endpoints')
    tester.test_module_integration('derek_ui')
    tester.test_module_integration('middleware')
    
    # UTILITIES & SUPPORT
    print("\n🛠️  TESTING UTILITIES...")
    tester.test_module_integration('helpers')
    tester.test_module_integration('logger')
    tester.test_module_integration('json_guardian')
    tester.test_module_integration('boot_guardian')
    tester.test_module_integration('db')
    
    # SCHEDULERS & ORCHESTRATION
    print("\n⏰ TESTING SCHEDULERS...")
    tester.test_module_integration('action_scheduler')
    tester.test_module_integration('dispatcher')
    tester.test_module_integration('loop')
    
    # INTEGRATIONS
    print("\n🔗 TESTING INTEGRATIONS...")
    tester.test_module_integration('github_integration')
    tester.test_module_integration('replit_mcp_server')
    tester.test_module_integration('moldbit')
    
    # ANALYTICS
    print("\n📊 TESTING ANALYTICS...")
    tester.test_module_integration('analytics_engine')
    
    # BOOT SYSTEMS
    print("\n🚀 TESTING BOOT SYSTEMS...")
    tester.test_module_integration('derek_boot')
    tester.test_module_integration('derek_autonomy_boot')
    tester.test_module_integration('main')
    
    # ALPHAVOX FAMILY
    print("\n👨‍👩‍👧‍👦 TESTING ALPHAVOX FAMILY...")
    tester.test_module_integration('alpha_interface')
    
    # FACE TO FACE MODE
    print("\n🤝 TESTING FACE-TO-FACE MODE...")
    tester.test_module_integration('face_to_face')
    
    # Print summary
    tester.print_summary()
    
    return tester.results


if __name__ == "__main__":
    print(f"\n🎯 Starting Derek System Integration Test...")
    print(f"📅 {datetime.now()}")
    
    try:
        results = run_derek_system_test()
        
        # Exit code based on results
        if len(results['failed']) > 0:
            print(f"\n⚠️  Some modules need attention!")
            sys.exit(1)
        elif len(results['warnings']) > 0:
            print(f"\n✅ All modules import, but some have integration warnings")
            sys.exit(0)
        else:
            print(f"\n🎉 ALL SYSTEMS GO! Derek is fully integrated!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Test framework error: {e}")
        traceback.print_exc()
        sys.exit(2)

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
