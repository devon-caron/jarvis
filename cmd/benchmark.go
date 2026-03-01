package cmd

import (
	"fmt"
	"math/rand"
	"sync"

	"github.com/spf13/cobra"
)

var (
	bmSize      int
	bmWebSearch bool
	//	bmBatchMode    bool // This will always be enabled in benchmark mode
	bmSystemPrompt string
	bmMaxTokens    int
	bmTemperature  float64
	bmModelFlag    string
	bmGpuFlag      int
	bmPromptSize   string
)

var benchmarkCmd = &cobra.Command{
	Use:               "benchmark",
	Short:             "run a benchmark with a given set of prompt flags",
	Long:              `The Benchmark command allows users to test their systems with the set of prompt flags available with default prompting. This can be used to gauge performance, NVLink augmentations, and more,`,
	CompletionOptions: cobra.CompletionOptions{DisableDefaultCmd: true},
	RunE:              runBenchmark,
}

var smallPrompts = []string{
	"Summarize the French Revolution.",
	"Causes of World War I?",
	"Key outcomes of WWII?",
	"Significance of the Magna Carta?",
	"What caused the Black Death?",
	"Role of Martin Luther King Jr.?",
	"Main effects of the Industrial Revolution?",
	"Key events of the American Civil War?",
	"Significance of the Cuban Missile Crisis?",
	"Major causes of the Russian Revolution?",
	"Impact of the Renaissance?",
	"Key achievements of the Apollo program?",
	"Main reasons for the fall of Rome?",
	"Significance of the Battle of Hastings?",
	"Key outcomes of the Congress of Vienna?",
	"Causes of the Thirty Years' War?",
	"Main features of the Enlightenment?",
	"Role of Cleopatra in history?",
	"Key events of the Meiji Restoration?",
	"Significance of the Berlin Wall?",
	"Main causes of the Great Depression?",
	"Key outcomes of the Treaty of Versailles?",
	"Role of Joan of Arc?",
	"Main effects of the Columbian Exchange?",
	"Significance of the Battle of Midway?",
	"Key causes of the English Civil War?",
	"Main features of the Cold War?",
	"Significance of the Fall of Constantinople?",
	"Key outcomes of the Glorious Revolution?",
	"Main causes of the Haitian Revolution?",
	"Significance of the Battle of Stalingrad?",
	"Key features of the Silk Road?",
	"Main effects of the printing press?",
	"Significance of the Suez Crisis?",
	"Key events of the Iranian Revolution?",
	"Main causes of the Vietnam War?",
	"Significance of the Magna Carta?",
	"Key outcomes of the Treaty of Westphalia?",
	"Main features of the Byzantine Empire?",
	"Significance of the Battle of Waterloo?",
	"Key causes of the War of 1812?",
	"Main effects of the Dust Bowl?",
	"Significance of the D-Day landings?",
	"Key outcomes of the Yalta Conference?",
	"Main causes of the Korean War?",
	"Significance of the fall of the Berlin Wall?",
	"Key features of the Ottoman Empire?",
	"Main effects of the Opium Wars?",
	"Significance of the Battle of Marathon?",
	"Key outcomes of the First Crusade?",
	"Main causes of the Peasants' Revolt?",
	"Significance of the Edict of Nantes?",
}

var mediumPrompts = []string{
	"In your own words, summarize the main causes and outcome of the French Revolution. Who was affected, and how did it change France's government? Include key events and long-term impacts.",
	"Explain the causes, major battles, and consequences of the American Civil War. How did it reshape U.S. society, politics, and the institution of slavery in the 19th century?",
	"Describe the rise and fall of the Roman Empire. What internal and external factors contributed to its decline, and what legacy did it leave for later European civilizations?",
	"What were the key causes, major turning points, and global consequences of World War I? How did it reshape the political map and international relations after 1918?",
	"Analyze the causes, key events, and global impact of World War II. How did it alter the balance of power and lead to the Cold War between 1939 and 1945?",
	"Summarize the main ideas of the Enlightenment and explain how they influenced revolutionary movements in America, France, and Latin America during the late 18th and early 19th centuries.",
	"Describe the origins, key leaders, and global consequences of the Cold War. How did ideological conflict between the U.S. and USSR shape international relations from 1947 to 1991?",
	"Explain the causes, major events, and long-term effects of the Russian Revolution. How did it transform Russia's political system, economy, and society under Bolshevik rule?",
	"Summarize the main causes, key figures, and outcomes of the Civil Rights Movement in the United States. How did it transform American law, society, and political participation?",
	"Describe the causes, key events, and global implications of the Industrial Revolution. How did it transform economies, societies, and daily life in Europe and North America?",
	"Analyze the causes, major phases, and consequences of the Protestant Reformation. How did it change religion, politics, and education across Europe in the 16th century?",
	"Summarize the main causes, key events, and long-term effects of the Hundred Years' War. How did it reshape England and France politically and militarily by 1453?",
	"Explain the causes, major developments, and global consequences of the Scientific Revolution. How did it transform understanding of nature and challenge traditional authority?",
	"Describe the causes, key events, and outcomes of the Glorious Revolution of 1688. How did it establish constitutional monarchy and influence later democratic movements?",
	"Analyze the causes, key leaders, and global impact of the Haitian Revolution. How did it transform slavery, colonialism, and human rights in the Atlantic world?",
	"Summarize the main causes, major events, and consequences of the Opium Wars. How did they reshape China's relationship with the West and its sovereignty in the 19th century?",
	"Explain the causes, key phases, and global consequences of the decolonization movement after 1945. How did it transform Africa, Asia, and the global power structure?",
	"Describe the causes, key events, and long-term effects of the Thirty Years' War. How did it reshape European politics, religion, and international law after 1648?",
	"Analyze the causes, major reforms, and consequences of the Meiji Restoration in Japan. How did it transform Japan into a modern industrial and military power by 1900?",
	"Summarize the main causes, key events, and global consequences of the Cuban Missile Crisis. How did it change U.S.-Soviet relations and nuclear policy during the Cold War?",
	"Explain the causes, major developments, and long-term effects of the Atlantic Slave Trade. How did it reshape African societies, the Americas, and global economics between 1500 and 1850?",
	"Describe the causes, key events, and outcomes of the English Civil War. How did it transform Britain's political system and lead to the execution of Charles I in 1649?",
	"Analyze the causes, key figures, and global consequences of the Mexican Revolution. How did it reshape Mexico's government, land ownership, and national identity after 1910?",
	"Summarize the main causes, major events, and long-term effects of the Spanish-American War. How did it transform the United States into a global imperial power in 1898?",
	"Explain the causes, key developments, and global impact of the Green Revolution. How did it transform agriculture, food production, and population dynamics in the 20th century?",
	"Explain the key events and long-term consequences of the American Civil War. How did it reshape the nation's political and social landscape?",
	"Describe the main causes and global effects of World War I. Which empires collapsed, and how did the war redrew European borders?",
	"What were the primary factors leading to World War II, and how did it transform international relations and global power structures afterward?",
	"Summarize the main causes, key events, and ultimate outcome of the Russian Revolution. How did it establish Soviet rule and impact global communism?",
	"Explain the origins, major events, and long-term impact of the Cold War. How did it shape international alliances and global politics for decades?",
	"Analyze the causes, key figures, and outcomes of the Civil Rights Movement in the United States. How did it transform laws and societal attitudes?",
	"Describe the causes, major phases, and global consequences of the Industrial Revolution. How did it change work, cities, and daily life?",
	"Summarize the main events and lasting impacts of the colonization of Africa. How did it affect African societies and global power dynamics?",
	"Explain the causes, key events, and outcomes of the German unification in the 19th century. How did it alter the balance of power in Europe?",
	"Describe the causes, major turning points, and consequences of the Thirty Years' War. How did it reshape religious and political authority in Europe?",
	"What were the causes and global consequences of the Opium Wars? How did they affect China's sovereignty and its relations with Western powers?",
	"Summarize the causes, key events, and outcomes of the Haitian Revolution. Why was it unique among revolutionary movements of its time?",
	"Explain the causes, major events, and global implications of the Iranian Revolution of 1979. How did it transform Iran's political and religious identity?",
	"Describe the causes, key figures, and outcomes of the Indian Independence Movement. How did nonviolent resistance shape the path to independence?",
	"Summarize the main causes, turning points, and effects of the Protestant Reformation. How did it alter religion, politics, and education in Europe?",
	"Explain the causes, major events, and consequences of the Meiji Restoration in Japan. How did it transform Japan into a modern industrial power?",
	"Describe the causes, key developments, and global impact of the Arab Spring. How did it reshape politics across the Middle East and North Africa?",
	"What were the causes, key figures, and outcomes of the Glorious Revolution of 1688? How did it influence constitutional government in Britain and beyond?",
	"Summarize the main causes, major battles, and consequences of the Peloponnesian War. How did it reshape ancient Greek politics and culture?",
	"Explain the causes, key events, and long-term effects of the Mongol invasions. How did they reshape Eurasian trade, culture, and governance?",
	"Describe the causes, major phases, and consequences of the Spanish Civil War. How did it foreshadow World War II and influence global ideology?",
	"What were the main causes and outcomes of the Berlin Crisis of 1948–49? How did it intensify Cold War tensions and shape European security?",
	"Explain the causes, key events, and global significance of the Cuban Missile Crisis. How close did the world come to nuclear war?",
	"Summarize the causes, major events, and outcomes of the Iranian Hostage Crisis. How did it reshape U.S.-Iran relations and American politics?",
	"Describe the causes, key leaders, and long-term effects of the Anti-Apartheid Movement. How did international pressure contribute to change in South Africa?",
	"What were the main causes and consequences of the Dust Bowl? How did it reshape U.S. agricultural policy and migration patterns?",
	"Explain the causes, major turning points, and outcomes of the Battle of Stalingrad. Why was it considered the turning point of the Eastern Front?",
	"Summarize the causes, key events, and lasting effects of the Partition of India in 1947. How did it reshape South Asian geopolitics and demographics?",
	"Describe the causes, major developments, and consequences of the Space Race. How did it influence technology, education, and international prestige?",
	"What were the main causes and global implications of the 2008 Global Financial Crisis? How did it reshape economic policy and public trust in institutions?",
}

var largePrompts = []string{
	"Analyze the American Civil Rights Movement by examining its origins, pivotal events, and lasting impacts. How did Jim Crow laws, the Montgomery Bus Boycott, the March on Washington, the Civil Rights Act, and the Voting Rights Act catalyze social change, and what roles did Martin Luther King Jr, Rosa Parks, Malcolm X, and other key figures play in shaping the movement's trajectory? Evaluate the movement's influence on modern social justice movements and its enduring legacy on American society.",
	"Analyze the Russian Revolution by examining its root causes, pivotal events, and far-reaching consequences. How did economic hardship, World War I, the February Revolution, the October Revolution, and the Russian Civil War contribute to the overthrow of the Romanov dynasty, and what roles did Vladimir Lenin, Leon Trotsky, and Joseph Stalin play in shaping the Soviet Union's early years, and evaluate the revolution's impact on modern communism, its influence on the Cold War, and its enduring legacy on Russian politics and society.",
	"Analyze the French Revolution by examining its causes, key events, and consequences. How did economic crisis, social inequality, and Enlightenment ideas contribute to the outbreak of the revolution, and what roles did Maximilien Robespierre, Napoleon Bonaparte, and other key figures play in shaping the revolution's trajectory? Evaluate the revolution's impact on modern democracy, its influence on European politics, and its enduring legacy on French society.",
	"Analyze the Industrial Revolution by examining its origins, key developments, and global impact. How did technological innovations, agricultural improvements, and urbanization transform society, and what roles did key inventors and industrialists play in shaping the era? Evaluate the revolution's impact on modern industrial society, its influence on labor movements, and its enduring legacy on global economic development.",
	"Analyze the Cold War by examining its origins, key events, and global impact. How did the ideological conflict between the United States and Soviet Union shape international relations, and what roles did key leaders like Truman, Kennedy, Khrushchev, and Gorbachev play in shaping the era? Evaluate the war's impact on global politics, its influence on nuclear proliferation, and its enduring legacy on international relations.",
	"Analyze the World Wars by examining their causes, key battles, and global consequences. How did World War I's trench warfare and World War II's global scale reshape international order, and what roles did key leaders like Churchill, Roosevelt, and Hitler play in shaping the 20th century? Evaluate the wars' impact on modern international law, their influence on the Cold War, and their enduring legacy on global security.",
	"Analyze the Renaissance by examining its origins, key figures, and cultural impact. How did the revival of classical learning and humanism transform European society, and what roles did Leonardo da Vinci, Michelangelo, and other key figures play in shaping the era? Evaluate the Renaissance's impact on art, science, and philosophy, and its enduring legacy on Western civilization.",
	"Analyze the Enlightenment by examining its philosophical foundations, key thinkers, and societal impact. How did ideas of reason, individualism, and skepticism challenge traditional authority, and what roles did Voltaire, Rousseau, and other key figures play in shaping the era? Evaluate the Enlightenment's impact on modern democracy, human rights, and scientific inquiry, and its enduring legacy on contemporary thought.",
	"Analyze the Scientific Revolution by examining its origins, key figures, and impact on modern science. How did the shift from medieval to modern scientific thinking transform human understanding, and what roles did Galileo, Newton, and other key figures play in shaping the era? Evaluate the revolution's impact on modern scientific methodology, its influence on the Enlightenment, and its enduring legacy on contemporary science.",
	"Analyze the Age of Exploration by examining its causes, key voyages, and global impact. How did European expansion reshape world trade and cultural exchange, and what roles did Columbus, da Gama, and other key explorers play in shaping the era? Evaluate the age's impact on global colonization, its influence on modern globalization, and its enduring legacy on international relations.",
	"Analyze the Protestant Reformation by examining its causes, key figures, and religious impact. How did Martin Luther's challenge to the Catholic Church spark religious reform, and what roles did John Calvin and other key figures play in shaping the era? Evaluate the reformation's impact on religious diversity, its influence on the Enlightenment, and its enduring legacy on modern Christianity.",
	"Analyze the Black Death by examining its causes, spread, and societal impact. How did the pandemic reshape European society, economy, and culture, and what roles did medical knowledge and social structures play in mitigating its effects? Evaluate the plague's impact on labor systems, its influence on the Renaissance, and its enduring legacy on public health.",
}

func init() {
	benchmarkCmd.Flags().IntVar(&bmSize, "size", 10, "Total number of prompts sent in benchmark")
	benchmarkCmd.Flags().StringVar(&bmPromptSize, "prompt-size", "small", "Size of each prompt (small, medium, large, varied). Uses a set of historical questions for consistent benchmarking.")
	benchmarkCmd.Flags().BoolVarP(&bmWebSearch, "web", "w", false, "Augment prompt with web search results")
	benchmarkCmd.Flags().StringVar(&bmSystemPrompt, "system", "", "Override system prompt")
	benchmarkCmd.Flags().IntVarP(&bmMaxTokens, "max-tokens", "n", 0, "Max tokens to generate (0 = config default)")
	benchmarkCmd.Flags().Float64VarP(&bmTemperature, "temperature", "t", 0, "Temperature (0 = config default)")
	benchmarkCmd.Flags().StringVarP(&bmModelFlag, "model", "m", "", "Target model name (when multiple models loaded)")
	benchmarkCmd.Flags().IntVarP(&bmGpuFlag, "gpu", "g", -1, "Route to whichever model is loaded on this GPU")
	rootCmd.AddCommand(benchmarkCmd)
}

func runBenchmark(cmd *cobra.Command, args []string) error {
	cmd.SilenceUsage = true
	var wg sync.WaitGroup

	var reports []*StatsReport
	for i := range bmSize {
		prompt := genRandPrompt(bmPromptSize)
		wg.Go(func() {
			fmt.Printf("Run %d: %s\n", i+1, prompt[:min(50, len(prompt))])
			response, statsReport, err := handleChat(prompt, Flags{
				WebSearch:    bmWebSearch,
				BatchMode:    true,
				SystemPrompt: bmSystemPrompt,
				MaxTokens:    bmMaxTokens,
				Temperature:  bmTemperature,
				ModelFlag:    bmModelFlag,
				GPUFlag:      bmGpuFlag,
				StatsFlag:    true,
			}, true)
			if err != nil {
				fmt.Printf("Error in benchmark run: %v\n", err)
				// no need to return
			}
			fmt.Printf("Run %d completed: %s\n", i+1, response[:min(50, len(response))])
			reports = append(reports, statsReport)
		})
	}
	wg.Wait()
	fmt.Println("stats reports:", len(reports))
	var totalTPS float64
	var totalTokens int64
	for i, report := range reports {
		fmt.Printf("Run %d: Tokens=%d, TPS=%f\n", i+1, report.TotalTokens, report.TokensPerSecond)
		totalTPS += report.TokensPerSecond
		totalTokens += int64(report.TotalTokens)
	}
	fmt.Printf("Average TPS: %f\n", totalTPS/float64(len(reports)))
	fmt.Printf("Total tokens: %d\n", totalTokens)
	fmt.Printf("Average tokens per run: %f\n", float64(totalTokens)/float64(len(reports)))
	fmt.Printf("Average TPS per run: %f\n", totalTPS/float64(len(reports)))
	fmt.Printf("Best TPS: %f\n", getMaxTPS(reports))
	fmt.Printf("Worst TPS: %f\n", getMinTPS(reports))
	return nil
}

func getMaxTPS(reports []*StatsReport) float64 {
	max := reports[0].TokensPerSecond
	for _, report := range reports {
		if report.TokensPerSecond > max {
			max = report.TokensPerSecond
		}
	}
	return max
}

func getMinTPS(reports []*StatsReport) float64 {
	min := reports[0].TokensPerSecond
	for _, report := range reports {
		if report.TokensPerSecond < min {
			min = report.TokensPerSecond
		}
	}
	return min
}

func genRandPrompt(size string) string {
	switch size {
	case "small":
		return smallPrompts[rand.Intn(len(smallPrompts))]
	case "medium":
		return mediumPrompts[rand.Intn(len(mediumPrompts))]
	case "large":
		return largePrompts[rand.Intn(len(largePrompts))]
	case "varied":
		randSize := rand.Intn(3)
		var promptSet []string
		switch randSize {
		case 0:
			promptSet = smallPrompts
		case 1:
			promptSet = mediumPrompts
		case 2:
			promptSet = largePrompts
		}
		return promptSet[rand.Intn(len(promptSet))]
	default:
		return "Hello"
	}
}
