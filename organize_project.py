#!/usr/bin/env python3
"""
Sunny Project Organizer
-----------------------
Organizes the Sunny project into a clean, professional structure
"""

import os
import shutil
from pathlib import Path

def organize_derek_project():
    """Organize Sunny project files into proper structure"""
    
    # Define the new structure based on your module categories
    file_mappings = {
        # Core modules by category
        'src/modules/consciousness': [
            'brain.py', 'sunny_identity.py', 'local_reasoning_engine.py', 
            'reasoning_engine.py', 'cognitive_bridge.py'
        ],
        'src/modules/memory': [
            'memory_engine.py', 'memory_manager.py', 'memory_router.py',
            'memory_hook.py', 'memory.py', 'memory_backup.py', 'memory_mesh_bridge.py'
        ],
        'src/modules/learning': [
            'ai_learning_engine.py', 'advanced_learning.py', 'sunny_learning_coordinator.py',
            'learning_analytics.py', 'learning_utils.py', 'knowledge_engine.py',
            'knowledge_integration.py', 'sunny_knowledge_engine.py', 'autonomous_learning_engine.py'
        ],
        'src/modules/emotion': [
            'tone_manager.py', 'emotion.py', 'behavioral_interpreter.py',
            'behaviors_interpreter.py', 'behavior_capturer.py', 'adaptive_conversation.py',
            'emotion_tagging.py'
        ],
        'src/modules/temporal': [
            'engine_temporal.py', 'alphavox_temporal.py', 'audio_pattern_service.py'
        ],
        'src/modules/vision': [
            'vision_engine.py', 'facial_gesture_service.py', 'real_eye_tracking.py',
            'eye_tracking_api.py'
        ],
        'src/modules/speech': [
            'advanced_tts_service.py', 'alphavox_speech_module.py', 
            'enhanced_speech_recognition.py', 'real_speech_recognition.py',
            'audio_processor.py', 'voice_analysis_service.py', 'sunny_ultimate_voice.py',
            'tts_service.py', 'tts_bridge.py', 'speech_recognition_engine.py'
        ],
        'src/modules/gesture': [
            'gesture_manager.py', 'gesture_dictionary.py', 'nonverbal_expertiser.py'
        ],
        'src/modules/conversation': [
            'conversation_engine.py', 'conversation_bridge.py', 'conversation_loop.py',
            'conversation_old.py'
        ],
        'src/modules/language': [
            'language_service.py', 'nlp_module.py', 'nlp_integration.py', 'nlu_core.py'
        ],
        'src/modules/internet': [
            'internet_mode.py', 'Python_Internet_access.py', 'perplexity_service.py',
            'web_crawler.py'
        ],
        'src/modules/reasoning': [
            'intent_engine.py', 'reflective_planner.py', 'input_analyzer.py'
        ],
        'src/modules/autonomous': [
            'sunny_autonomous_system.py', 'self_modifying_code.py', 'executor.py',
            'interpreter.py'
        ],
        'src/modules/web': [
            'app.py', 'api.py', 'endpoints.py', 'sunny_ui.py', 'middleware.py',
            'router.py', 'backenddirect.py', 'sunnydirect.py'
        ],
        'src/modules/utilities': [
            'helpers.py', 'logger.py', 'json_guardian.py', 'boot_guardian.py',
            'db.py', 'database.py', 'check_env.py'
        ],
        'src/modules/scheduling': [
            'action_scheduler.py', 'dispatcher.py', 'loop.py'
        ],
        'src/modules/integrations': [
            'github_integration.py', 'moldbit.py'
        ],
        'src/modules/analytics': [
            'analytics_engine.py'
        ],
        'src/modules/interaction': [
            'face_to_face.py', 'hotline.py'
        ],
        
        # Configuration files
        'config': [
            '*.json', '*.yaml', '*.yml', 'config.py', 'settings.py',
            'mcp_config.json', 'sunny_manifest.yaml', 'sunny-manifest.yaml'
        ],
        
        # Data files
        'data': [
            'sunny_memory.json', 'knowledge_graph.json', 'facts.json',
            'curriculum.json', 'learning_chambers.json', 'language_map.json',
            'improvement_suggestions.json', 'sunny_identity.json'
        ],
        
        # Documentation
        'docs': [
            '*.md', '*.txt', '*.ipynb', 'DEREK_*.txt', 'BROKEN_MODULES_LIST.txt'
        ],
        
        # Scripts
        'scripts': [
            '*.sh', 'install.sh', 'activate_venv.sh', 'GATHER_DEREKS_MEMORY.sh',
            'start_derek.sh'
        ],
        
        # Logs
        'logs': [
            '*.log', '*.jsonl', 'internet_log.jsonl', 'internet_log.txt',
            'crawler_status.json'
        ],
        
        # Web assets
        'web': [
            'index.html', '*.js', '*.css'
        ],
        
        # Boot/Core files (stay in root)
        '.': [
            'main.py', 'sunny_boot.py', 'sunny_autonomy_boot.py', 
            'sunny_module_loader.py', 'core.py', 'run.py'
        ]
    }
    
    print("🗂️  Organizing Sunny Project Structure")
    print("=" * 50)
    
    # Create directories if they don't exist
    for directory in file_mappings.keys():
        if directory != '.':
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Created: {directory}")
    
    # Move files to appropriate directories
    moved_count = 0
    for target_dir, patterns in file_mappings.items():
        for pattern in patterns:
            if '*' in pattern:
                # Handle wildcards
                import glob
                files = glob.glob(pattern)
                for file in files:
                    if os.path.isfile(file) and target_dir != '.':
                        try:
                            shutil.move(file, os.path.join(target_dir, os.path.basename(file)))
                            print(f"  📄 {file} → {target_dir}/")
                            moved_count += 1
                        except Exception as e:
                            print(f"  ❌ Failed to move {file}: {e}")
            else:
                # Handle specific files
                if os.path.isfile(pattern) and target_dir != '.':
                    try:
                        shutil.move(pattern, os.path.join(target_dir, pattern))
                        print(f"  📄 {pattern} → {target_dir}/")
                        moved_count += 1
                    except Exception as e:
                        print(f"  ❌ Failed to move {pattern}: {e}")
    
    print(f"\n✅ Moved {moved_count} files")
    print("🎯 Sunny project is now organized!")
    
    # Create __init__.py files for Python packages
    init_dirs = [
        'src', 'src/modules', 'src/modules/consciousness', 'src/modules/memory',
        'src/modules/learning', 'src/modules/emotion', 'src/modules/speech',
        'src/modules/conversation', 'src/modules/web'
    ]
    
    for init_dir in init_dirs:
        init_file = os.path.join(init_dir, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write(f'"""Sunny {init_dir.replace("/", ".")} module"""\n')
            print(f"📦 Created: {init_file}")

if __name__ == "__main__":
    organize_derek_project()