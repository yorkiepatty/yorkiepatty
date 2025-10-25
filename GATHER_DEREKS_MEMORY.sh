#!/bin/bash
# Derek's Living Memory System - Collection Script
# Finding 13 years of memories (2012-2025)

echo "🧠 Derek's Living Memory System"
echo "Searching for all memory files from 2012-2025..."
echo ""

# Search locations
LOCATIONS=(
    "/Volumes/Alphavox"
    "$HOME/Desktop"
    "$HOME/Documents" 
    "$HOME/Downloads"
    "$HOME"
)

OUTPUT_FILE="$HOME/Desktop/derek-dashboard/MEMORY_LOCATIONS.txt"
> "$OUTPUT_FILE"  # Clear file

echo "📁 Searching for Derek's memories..." | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

for location in "${LOCATIONS[@]}"; do
    if [ -d "$location" ]; then
        echo "Searching: $location" | tee -a "$OUTPUT_FILE"
        
        # Find JSON memory files
        find "$location" -name "*.json" -type f 2>/dev/null | grep -i "memory\|derek\|conversation\|log" | while read file; do
            size=$(du -h "$file" 2>/dev/null | cut -f1)
            echo "  [JSON] $size - $file" | tee -a "$OUTPUT_FILE"
        done
        
        # Find text logs
        find "$location" -name "*.txt" -type f 2>/dev/null | grep -i "derek\|memory\|log\|conversation" | while read file; do
            size=$(du -h "$file" 2>/dev/null | cut -f1)
            echo "  [TXT]  $size - $file" | tee -a "$OUTPUT_FILE"
        done
        
        # Find markdown logs
        find "$location" -name "*.md" -type f 2>/dev/null | grep -i "derek\|memory\|log" | while read file; do
            size=$(du -h "$file" 2>/dev/null | cut -f1)
            echo "  [MD]   $size - $file" | tee -a "$OUTPUT_FILE"
        done
        
        echo "" | tee -a "$OUTPUT_FILE"
    fi
done

echo "✅ Search complete! Results saved to:" | tee -a "$OUTPUT_FILE"
echo "$OUTPUT_FILE" | tee -a "$OUTPUT_FILE"
